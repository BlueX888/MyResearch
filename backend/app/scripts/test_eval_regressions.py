"""评测暴露的来源事件遗漏、审核结构异常回归；真实 HTTP 验证另由 eval 运行。"""
import json
import logging
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from service.deep_research_v2.agents.scout import DeepScout
from service.deep_research_v2.agents.critic import CriticMaster


class SearchEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scout = object.__new__(DeepScout)
        self.scout.name = "DeepScout"
        self.scout.logger = logging.getLogger("test.scout")
        self.events = []
        self.scout.add_message = lambda state, kind, content: self.events.append((kind, content))
        self.results = [{"url": f"https://source.example/{n}", "summary": f"原文{n}"} for n in range(8)]

    def assert_all_sources_sent(self):
        rows = [r for kind, content in self.events if kind == "search_results"
                and content.get("isIncremental") for r in content["results"]]
        self.assertEqual({r["url"] for r in rows}, {r["url"] for r in self.results})
        self.assertEqual({r["snippet"] for r in rows}, {r["summary"] for r in self.results})

    async def test_initial_search_sends_sources_after_fifth_result(self):
        for mode in ("web", "local"):
            with self.subTest(mode=mode):
                self.events.clear()
                self.scout._execute_search = AsyncMock(return_value=self.results)
                self.scout._execute_local_search = AsyncMock(return_value=self.results)
                self.scout._analyze_search_results = AsyncMock(return_value=None)
                await self.scout._research_section(
                    {"query": "问题", "search_web": mode == "web", "search_local": mode == "local"},
                    {"id": "section", "title": "章节", "search_queries": ["查询"]})
                self.assert_all_sources_sent()

    async def test_deep_search_sends_all_sources(self):
        self.scout._search_by_mode = AsyncMock(return_value=self.results)
        self.scout._analyze_deep_search_results = AsyncMock(return_value=None)
        await self.scout._execute_deep_search({"query": "问题"}, "section", ["查询"], "follow_up", [])
        self.assert_all_sources_sent()

    async def test_one_timed_out_section_cannot_leave_other_retrieval_running_into_writing(self):
        import asyncio
        state = {"phase": "researching", "search_web": False, "search_local": True, "facts": [],
                 "outline": [{"title": t, "status": "pending"} for t in ("甲", "乙")]}
        async def retrieve(state, section):
            if section["title"] == "甲":
                raise TimeoutError()
            await asyncio.sleep(0.01)
            self.scout.record_results(state, [{"url":"local://kb/doc", "summary":"乙的原文"}], "local")
        self.scout._research_section = retrieve
        await self.scout.process(state)
        self.assertEqual(state["raw_sources"][0]["original_text"], "乙的原文")
        self.assertEqual(len(state["errors"]), 1)

    async def test_supplementary_search_sends_original_evidence(self):
        self.scout._search_by_mode = AsyncMock(return_value=self.results)
        self.scout._analyze_supplementary_results = AsyncMock(return_value=None)
        await self.scout._supplementary_research({"query": "问题", "pending_search_queries": ["查询"], "facts": []})
        self.assert_all_sources_sent()


class ReviewSchemaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.critic = object.__new__(CriticMaster)
        self.critic.logger = logging.getLogger("test.critic")
        self.state = {"query": "问题", "draft_sections": {}, "final_report": "报告", "facts": [],
                      "data_points": [], "outline": []}
        self.valid = {"overall_assessment": {"verdict": "major_issues", "quality_score": 3},
                      "issues": [{"severity": "major", "issue_type": "logic_error", "description": "问题"}],
                      "missing_aspects": []}

    async def test_string_issue_is_retried_before_entering_state(self):
        invalid = {**self.valid, "issues": ["字符串问题"]}
        self.critic.call_llm = AsyncMock(side_effect=[json.dumps(invalid), json.dumps(self.valid)])
        self.assertEqual(await self.critic._review_content(self.state), self.valid)
        self.assertEqual(self.critic.call_llm.await_count, 2)

    async def test_repeated_invalid_review_cannot_pass(self):
        self.critic.call_llm = AsyncMock(return_value=json.dumps({**self.valid, "issues": "不是对象数组"}))
        self.assertIsNone(await self.critic._review_content(self.state))
        self.assertEqual(self.critic.call_llm.await_count, 2)


class FinalPublicationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.url = "local://kb/report"
        self.row = {"url": self.url, "name": "原始年报", "summary": "甲公司营收12元。", "source": "local"}

    async def check_numbers(self, system, prompt):
        jobs = json.loads(prompt)["items"]
        return json.dumps({"items": [{"i": j["i"], "p": j["candidates"][0],
                                     "subject": True, "value": True, "unit": True, "time": True} for j in jobs]})

    async def test_classic_lookup_without_a_kb_name_stays_local(self):
        import asyncio
        from service.dr_g import parallel_search_all
        with patch('service.dr_g.search_local_knowledge', AsyncMock(return_value=[self.row])) as local, patch('service.dr_g.websearch') as web:
            found = await parallel_search_all(['财报原文'], asyncio.Semaphore(1), None, False, True)
        self.assertEqual(found, [('财报原文', [self.row], 'local')])
        local.assert_awaited_once()
        web.assert_not_called()

    def test_completed_checkpoint_commits_audited_text_with_status(self):
        from service.checkpoint_service import CheckpointService
        service = object.__new__(CheckpointService)
        db = MagicMock()
        existing = db.query.return_value.filter.return_value.first.return_value
        service._get_db = lambda: db
        for audit, status in [({}, "running"), ({"numeric_audit":{"items":[]}}, "completed")]:
            state = {"query": "财报", "phase": "completed", "final_report": "正文", **audit}
            service.save_checkpoint("session", state, final_report="正文")
            self.assertEqual(existing.status, status)
            self.assertEqual(existing.final_report, existing.state_json["final_report"])
        self.assertEqual(db.commit.call_count, 2)

    async def test_v1_buffers_draft_and_looks_up_in_selected_mode(self):
        from service.dr_g import ResearchService
        service = object.__new__(ResearchService)
        service.use_react = True
        service.llm_api_key = "test"
        service.llm_base_url = "http://unused"
        service.tool_executor = SimpleNamespace(execute_web_search=AsyncMock(), execute_knowledge_search=AsyncMock(return_value=[]))
        client = MagicMock()
        chunks = ["甲公司营收12元。", "乙公司利润99元。"]
        client.chat.completions.create.return_value = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=s))]) for s in chunks]
        with patch("service.dr_g.OpenAI", return_value=client), patch("service.dr_g.model_verifier", return_value=self.check_numbers):
            events = [json.loads(e) async for e in service._generate_final_report(
                "利润与营收", [self.row], [], [], "财报库", False, True)]
        kinds = [e["type"] for e in events]
        self.assertLess(kinds.index("numeric_audit"), kinds.index("answer_start"))
        self.assertLess(kinds.index("reference_materials"), kinds.index("answer"))
        published = "".join(e["content"] for e in events if e["type"] == "answer")
        self.assertIn("12元", published)
        self.assertNotIn("99", published)
        service.tool_executor.execute_web_search.assert_not_awaited()
        ctx = service.tool_executor.execute_knowledge_search.await_args.args[1]
        self.assertEqual(ctx.metadata["kb_name"], "财报库")

    async def test_v2_checks_last_revision_and_checkpoints_the_same_report(self):
        from report_numbers import source_record
        from service.deep_research_v2.graph import DeepResearchGraph
        from service.deep_research_v2.state import create_initial_state
        graph = object.__new__(DeepResearchGraph)
        graph.checkpoint_service = None
        state = create_initial_state("甲公司财报", "", False, True)
        state["max_iterations"] = 1

        async def search(state):
            state["raw_sources"] = [source_record(self.row)]
        async def write(state):
            state["final_report"] = "甲公司营收12元。" + ("乙公司利润99元。" if state["phase"]=="revising" else "")
        async def review(state):
            state.update(phase="revising", iteration=1)
        def agent(name, process):
            return SimpleNamespace(name=name, process=process)
        graph.architect = agent("规划", AsyncMock())
        graph.scout = agent("检索", search)
        graph.scout._search_by_mode = AsyncMock(return_value=[])
        graph.scout.record_results = MagicMock()
        graph.data_analyst = graph.wizard = agent("分析", AsyncMock())
        graph.writer = agent("写作", write)
        graph.critic = agent("审核", review)
        graph.critic.client = graph.critic.model = None
        saved = []
        def save(state, user_id, ui):
            saved.append((state["phase"], state["final_report"], ui["streaming_report"], ui["references"]))
            return True
        graph._save_checkpoint = save
        with patch("service.deep_research_v2.graph.model_verifier", return_value=self.check_numbers):
            events = [e async for e in graph._run_simplified(state)]
        final = events[-1]
        self.assertEqual(final["type"], "research_complete")
        self.assertIn("12元", final["final_report"])
        self.assertNotIn("99", final["final_report"])
        self.assertEqual(saved[-1], ("completed", final["final_report"], final["final_report"], final["references"]))
        self.assertEqual(state["numeric_audit"]["removed"][0]["text"], "99元")
        graph.scout._search_by_mode.assert_awaited_once()

    async def test_compiled_graph_also_finishes_with_verified_text(self):
        from report_numbers import source_record
        from service.deep_research_v2.graph import DeepResearchGraph, LANGGRAPH_AVAILABLE
        from service.deep_research_v2.state import create_initial_state
        if not LANGGRAPH_AVAILABLE:
            self.skipTest("LangGraph is optional")
        graph = object.__new__(DeepResearchGraph)
        async def passthrough(state):
            return state
        async def retrieve(state):
            state["raw_sources"] = [source_record(self.row)]
            return state
        async def write(state):
            state["final_report"] = "甲公司营收12元。"
            return state
        graph.architect = graph.wizard = SimpleNamespace(process=passthrough)
        graph.scout = SimpleNamespace(process=retrieve)
        graph.writer = SimpleNamespace(process=write)
        graph.critic = SimpleNamespace(process=passthrough, client=None, model=None)
        with patch("service.deep_research_v2.graph.model_verifier", return_value=self.check_numbers):
            out = await graph._build_langgraph().ainvoke(create_initial_state("甲营收", "test"))
        self.assertEqual(out["phase"], "completed")
        self.assertEqual(out["numeric_audit"]["supported"], 1)
        self.assertIn(self.url, out["final_report"])


if __name__ == "__main__":
    unittest.main()
