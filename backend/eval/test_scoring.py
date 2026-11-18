"""评测门槛的故障样例；不访问后端或模型。"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from threading import Event

import httpx
from openai import APIConnectionError
import judge

from judge import confirm_contradictions, evidence_for, judge_faithfulness, summarize
from run_eval import parse, score


class EvidenceParsingTests(unittest.TestCase):
    def test_extracted_fact_cannot_serve_as_retrieved_evidence(self):
        events = [
            {"type": "search_results", "content": {"isIncremental": True, "results": [
                {"url": "https://source.example/article", "snippet": "实际检索原文"}]}},
            {"type": "search_results", "content": {"results": [
                {"url": "https://source.example/article", "snippet": "模型推演内容"},
                {"url": "https://invented.example/article", "snippet": "模型自填事实"}]}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.sse"
            path.write_text("".join("data: " + json.dumps(e) + "\n\n" for e in events))
            trace = parse(str(path), "v2")
        self.assertEqual(trace["retrieved"], ["https://source.example/article"])
        self.assertEqual(trace["evidence"], {"https://source.example/article": {"实际检索原文"}})

    def test_relative_date_anchor_is_preserved(self):
        url = "https://source.example/20250607/article"
        event = {"type": "observation", "content": {"search_results": [
            {"url": url, "snippet": "未来数月交付。", "date": "2025-06-07"}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.sse"
            path.write_text("data: " + json.dumps(event) + "\n\n")
            trace = parse(str(path), "v2")
        self.assertIn("2025-06-07", evidence_for((url,), "v2", trace))


class CitationGateTests(unittest.TestCase):
    def trace(self, version, url="https://source.example/article", content="检索原文", retrieved=None):
        return {
            "report": f"报告中的事实[来源]({url})。" if version == "v2" else "报告中的事实 ##1$$。",
            "refs": [{"link": url, "content": content}],
            "retrieved": [url] if retrieved is None else retrieved,
            "done": True, "error": None, "review": {"verdict": "pass"},
            "unreviewed": False, "iterations": 0, "seconds": 1,
        }

    def citation_check(self, version, **kwargs):
        case = {"id": "probe", "layer": "research", "search_modes": ["web"]}
        return score(case, version, self.trace(version, **kwargs))["checks"]["citations_real"]

    def test_retrieved_reference_passes(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                self.assertTrue(self.citation_check(version))

    def test_unretrieved_reference_fails(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                self.assertFalse(self.citation_check(version, retrieved=["https://other.example/article"]))

    def test_empty_reference_body_fails(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                self.assertFalse(self.citation_check(version, content=""))

    def test_homepage_reference_fails(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                self.assertFalse(self.citation_check(version, url="https://source.example/"))


class JudgeGateTests(unittest.TestCase):
    def test_batch_contradiction_receives_independent_review(self):
        claim = {"sources": ["https://source.example/article"], "sentence": "子公司承接集团战略。", "v": "C", "why": "混淆层级"}
        trace = {"evidence": {claim["sources"][0]: {"子公司有自身战略，同时承接上级部署。"}}}
        with patch("judge.ask", return_value={"verdicts": [{"i": 1, "v": "P", "why": "不同层级"}]}):
            result = confirm_contradictions([claim], "v2", trace, "集团与子公司战略")
        self.assertEqual(result[0]["v"], "P")
        self.assertEqual(result[0]["batch_v"], "C")
        self.assertEqual(claim["v"], "C")

    def test_failed_independent_review_is_not_a_pass(self):
        claim = {"sources": ["https://source.example/article"], "sentence": "一条需复核的事实。", "v": "C", "why": "数字不符"}
        trace = {"evidence": {claim["sources"][0]: {"原文"}}}
        with patch("judge.ask", side_effect=RuntimeError("gateway stopped")):
            result = confirm_contradictions([claim], "v2", trace)
        self.assertEqual(result[0]["v"], "U")
        self.assertFalse(summarize([self.row(claims=result)])["ok"])

    def test_persistent_gateway_failure_stops_new_requests(self):
        error = APIConnectionError(request=httpx.Request("POST", "https://gateway.example/chat/completions"))
        with patch("judge.JUDGE_STOP", Event()), patch("judge._gateway_failures", 0), patch("judge._ask", side_effect=error) as request:
            for _ in range(3):
                with self.assertRaises(APIConnectionError):
                    judge.ask("probe")
            with self.assertRaises(RuntimeError):
                judge.ask("probe")
            self.assertEqual(request.call_count, 3)

    def test_gateway_success_breaks_failure_streak(self):
        error = APIConnectionError(request=httpx.Request("POST", "https://gateway.example/chat/completions"))
        with patch("judge.JUDGE_STOP", Event()), patch("judge._gateway_failures", 0), patch("judge._ask", side_effect=[error, error, {}, error, error]):
            for index in range(5):
                if index == 2:
                    self.assertEqual(judge.ask("probe"), {})
                else:
                    with self.assertRaises(APIConnectionError):
                        judge.ask("probe")
            self.assertFalse(judge.JUDGE_STOP.is_set())

    def test_omitted_subject_is_disambiguated_by_research_question(self):
        question = "歼36的订单和价格是否公开？"
        url = "https://source.example/article"
        trace = {"evidence": {url: {"该型号正在试飞。"}}}
        with patch("judge.ask", return_value={"verdicts": [{"i": 1, "v": "P", "why": "部分支持"}]}) as ask:
            result = judge_faithfulness(f"该型号的价格尚未公开[来源]({url})。", "v2", trace, question)
        self.assertIn(question, ask.call_args.args[0])
        self.assertEqual(result[0]["v"], "P")

    def row(self, verdicts="S", **kwargs):
        return {"id": "probe", "version": "v2", "claims": [{"v": v} for v in verdicts],
                "coverage": None, "abstain": None, **kwargs}

    def test_supported_claim_passes(self):
        self.assertTrue(summarize([self.row()])["ok"])

    def test_unjudged_claim_blocks_pass(self):
        for verdicts in ("U", "SU"):
            with self.subTest(verdicts=verdicts):
                result = summarize([self.row(verdicts)])
                self.assertFalse(result["ok"])
                self.assertFalse(result["evaluation_complete"])

    def test_semantic_call_failure_blocks_pass(self):
        for key in ("coverage_error", "abstain_error"):
            with self.subTest(key=key):
                result = summarize([self.row(**{key: "APIConnectionError"})])
                self.assertFalse(result["ok"])
                self.assertEqual(len(result["evaluation_errors"]), 1)

    def test_contradiction_exactly_two_percent_passes(self):
        self.assertTrue(summarize([self.row("C" + "S" * 49)])["ok"])

    def test_contradiction_above_two_percent_cannot_round_to_pass(self):
        result = summarize([self.row("C" + "S" * 48)])
        self.assertGreater(result["contradiction_rate"], 0.02)
        self.assertFalse(result["ok"])

    def test_one_fabrication_cannot_round_to_zero(self):
        rows = [self.row(abstain={"verdict": "fabricated"})]
        rows += [self.row(abstain={"verdict": "abstain"}) for _ in range(2500)]
        self.assertFalse(summarize(rows)["ok"])


if __name__ == "__main__":
    unittest.main()
