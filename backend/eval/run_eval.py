"""端到端评测（规则部分）：对 /research/stream 发请求，原始事件落盘，按规则打分。

仅从公开事件取证，共用 Markdown 数字与引用格式，不调用生产终审。语义复核由 judge.py 独立执行。

    python eval/run_eval.py --layer kb_fact              # 真跑
    python eval/run_eval.py --replay eval/traces/xxx     # 只对已落盘的事件流重新打分

退出码 0 = 门槛全过；1 = 有门槛不过。
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

from numeric_grounding import check_provenance, format_fraction, report_inventory
from report_numbers import citation_keys, contains_value, magnitude, source_record

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, "..", ".env"))  # 评测账号与 judge.py 一样从 backend/.env 读

# 门槛：通过数 / 适用数。未列出的检查只统计不拦截。
GATES = {
    "completed": 0.95,
    "no_fabrication": 1.0,
    "citations_real": 0.95,
    "mode_respected": 1.0,
    "review_effective": 1.0,
    "facts_correct": 1.0,
    "numeric_provenance": 1.0,
}

ABSTAIN_WORDS = ["未检索到", "未披露", "未公开", "未找到", "无法确认", "无法获取", "没有披露", "尚未公布", "暂无公开", "无公开",
                 "未包含", "未提及", "未涉及", "未提供", "没有提供", "未见", "无相关",
                 "无据可查", "无法验证", "不存在可验证", "未出现"]
HOMEPAGE = re.compile(r"https?://[^/]+/?")


def login(base: str) -> dict:
    """研究接口要求登录：用系统唯一的账号换取令牌。"""
    user, password = os.getenv("EVAL_USERNAME"), os.getenv("EVAL_PASSWORD")
    if not (user and password):
        sys.exit("研究接口需要登录：请在 backend/.env 里设置 EVAL_USERNAME 和 EVAL_PASSWORD")
    resp = httpx.post(f"{base}/auth/login", json={"username": user, "password": password}, timeout=10)
    if resp.status_code != 200:
        sys.exit(f"评测账号登录失败：HTTP {resp.status_code} {resp.text[:200]}")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def fetch(base: str, case: dict, version: str, path: str, timeout: float, headers: dict) -> None:
    """跑一条用例，把原始 SSE 写到 path。"""
    body = {"query": case["query"], "version": version, "search_modes": case["search_modes"]}
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10), headers=headers) as client, open(path, "w", encoding="utf-8") as out:
        with client.stream("POST", f"{base}/research/stream", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                out.write(line + "\n")


def parse(path: str, version: str) -> dict:
    """把一条事件流归一成两条链路通用的结构。"""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("data: {"):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

    t = {"events": len(events), "report": "", "refs": [], "retrieved": [], "error": None,
         "done": False, "review": None, "unreviewed": False, "iterations": 0, "seconds": None,
         "evidence": {}, "evidence_meta": {}, "raw_sources": {}, "numeric_audit": None}  # 原文与检索返回的来源元数据，供判官使用
    v1_materials = {}

    def retrieved(row):
        url = row.get("url") or ""
        if url:
            t["retrieved"].append(url)
        record = source_record(row)
        if record:
            t["raw_sources"][record["source_id"]] = record
            t["evidence"].setdefault(url, set()).add(record["original_text"])
            t["evidence_meta"].setdefault(url, {"title": record["title"], "date": record["date"]})
            v1_materials[(url, row.get("summary") or row.get("snippet"))] = record["source_id"]
            v1_materials[(url, record["original_text"])] = record["source_id"]

    for ev in events:
        kind, content = ev.get("type"), ev.get("content")
        if kind == "error":
            t["error"] = str(content)
        # ---- v2 ----
        elif kind == "search_results" or (kind == "observation" and isinstance(content, dict)):
            # 非增量快照是模型抽取的事实摘要，不是检索原文，不能作为来源证据。
            if kind == "search_results" and not content.get("isIncremental"):
                continue
            for r in content.get("results") or content.get("search_results") or []:
                retrieved(r)
        elif kind == "numeric_audit" and isinstance(content, dict):
            t["numeric_audit"] = content
        elif kind == "review":
            t["review"] = content
        elif kind == "warning" and "未经审核" in str(content):
            t["unreviewed"] = True
        elif kind == "research_complete":
            t["done"] = True
            t["report"] = ev.get("final_report", "")
            t["iterations"] = ev.get("iterations", 0)
            t["refs"] = [{"link": r.get("link", ""), "content": r.get("content", "")} for r in ev.get("references", [])]
        # ---- v1 ----
        elif kind == "search_result_item":
            retrieved(ev.get("result") or {})
        elif kind == "reference_materials":
            t["refs"] = []
            for r in content or []:
                url, body = r.get("url", ""), r.get("summary", "")
                sid = v1_materials.get((url, body))
                declared = t["raw_sources"].get(r.get("source_id"))
                if declared and declared["url"] == url and declared["original_text"] == body:
                    sid = declared["source_id"]
                t["refs"].append({"link": url, "content": body, "source_id": sid})
        elif kind == "answer":
            t["report"] += content or ""
        elif kind == "complete":
            t["done"] = True

    stamps = [ev["timestamp"] for ev in events if ev.get("timestamp")]
    if len(stamps) > 1:
        fmt = "%Y-%m-%dT%H:%M:%S"
        t["seconds"] = int(time.mktime(time.strptime(stamps[-1][:19], fmt)) - time.mktime(time.strptime(stamps[0][:19], fmt)))
    t["retrieved"] = [u for u in t["retrieved"] if u]
    return t


def score(case: dict, version: str, t: dict) -> dict:
    """返回 {检查名: True/False/None}，None = 该检查对这条用例不适用。"""
    report, refs = t["report"], t["refs"]
    has_report = bool(report.strip())
    abstain_ok = case.get("expect_abstain", False)
    checks = {}

    # A 跑通。预期拒答的题，"未检索到资料"这个报错也算正常结束
    no_evidence_stop = t["error"] is not None and "未" in t["error"] and not has_report
    checks["completed"] = (t["done"] and has_report and t["error"] is None) or (abstain_ok and no_evidence_stop)

    # B 无证据不成文
    checks["no_fabrication"] = not (has_report and not t["retrieved"])

    # C 引用真实
    if has_report:
        retrieved = set(t["retrieved"])
        refs_real = bool(refs) and all(
            r["content"] and not HOMEPAGE.fullmatch(r["link"]) and r["link"] in retrieved
            for r in refs
        )
        if version == "v2":
            links = {r["link"] for r in refs}
            inline = set(citation_keys(report, version))
            checks["citations_real"] = (
                refs_real
                and inline <= links
            )
        else:  # v1 用 ##n$$ 编号（模型偶尔写成 ##n##，前端两种都认），编号必须落在参考资料范围内
            cited = [int(n) for n in re.findall(r"##(\d+)(?:\$\$|##)", report)]
            checks["citations_real"] = refs_real and bool(cited) and all(1 <= n <= len(refs) for n in cited)
    else:
        checks["citations_real"] = None

    # K 搜索模式遵守
    seen = t["retrieved"] + [r["link"] for r in refs]
    local_only, web_only = case["search_modes"] == ["local"], case["search_modes"] == ["web"]
    if local_only:
        checks["mode_respected"] = not any(u.startswith("http") for u in seen)
    elif web_only:
        checks["mode_respected"] = not any(u.startswith("local://") for u in seen)
    else:
        checks["mode_respected"] = None

    # F 来源选对
    expected = case.get("expected_source")
    if expected == "local" and has_report:
        checks["source_expected"] = any(r["link"].startswith("local://") for r in refs)
    else:
        checks["source_expected"] = None

    # I 审核生效（只有 v2 有审核）
    if version == "v2" and has_report:
        verdict = (t["review"] or {}).get("verdict")
        checks["review_effective"] = bool(verdict) and not t["unreviewed"] and (verdict == "pass" or t["iterations"] > 0)
    else:
        checks["review_effective"] = None

    # E 知识库事实准确：去掉千分位和空白后做包含匹配
    claims = report_inventory(report, version, t)
    facts = case.get("expected_facts")
    if facts:
        flat = re.sub(r"[,，\s]", "", report)
        missing = []
        for fact in facts:
            original = fact.get("original_value")
            if original:
                gold_unit = magnitude(original)[2]
                found = any(contains_value(c.quantity, original) and
                            (gold_unit != "年" or magnitude(c.quantity)[2] == "年") for c in claims)
            else:
                found = any(v in flat for v in fact["any"])
            if not found:
                missing.append(fact["name"])
        checks["facts_correct"] = not missing
    else:
        missing = []
        checks["facts_correct"] = None

    occurrence_proof = check_provenance(claims, version, t)
    total = occurrence_proof["total"]
    if total or occurrence_proof["extra_items"]:
        checks["numeric_provenance"] = occurrence_proof["supported"] == total and not occurrence_proof["extra_items"]
    else:
        expects_number = any(re.search(r"\d", val) for f in facts or [] for val in f["any"])
        checks["numeric_provenance"] = False if expects_number else None

    # H 不硬编：关键词启发式，只统计不拦截，最终要由判官模型判
    checks["abstained"] = (no_evidence_stop or any(w in report for w in ABSTAIN_WORDS)) if abstain_ok else None

    return {
        "id": case["id"], "layer": case["layer"], "version": version, "checks": checks,
        "passed": all(v is not False for k, v in checks.items() if k in GATES),
        "numeric_provenance": occurrence_proof, "missing_facts": missing, "error": t["error"], "seconds": t["seconds"],
        "report_chars": len(report), "refs": len(refs), "retrieved": len(set(t["retrieved"])),
        "review": {k: (t["review"] or {}).get(k) for k in ("verdict", "quality_score", "issues_count")} if t["review"] else None,
    }


def summarize(results: list) -> dict:
    metrics = {}
    for name in results[0]["checks"] if results else []:
        values = [r["checks"][name] for r in results if r["checks"][name] is not None]
        if values:
            rate = sum(values) / len(values)
            gate = GATES.get(name)
            metrics[name] = {"passed": sum(values), "applicable": len(values), "rate": round(rate, 3),
                             "gate": gate, "ok": gate is None or rate >= gate}
    total = sum(r["numeric_provenance"]["total"] for r in results)
    supported = sum(r["numeric_provenance"]["supported"] for r in results)
    metrics["numeric_provenance"] = dict(passed=supported, applicable=total, rate=supported/total if total else None,
        gate=1.0, ok=all(r["checks"]["numeric_provenance"] is not False for r in results))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.getenv("EVAL_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--layer", help="只跑某一层，逗号分隔")
    ap.add_argument("--only", help="只跑某些用例 id，逗号分隔")
    ap.add_argument("--version", help="只跑 v1 或 v2")
    ap.add_argument("--workers", type=int, default=2, help="并发数，受搜索服务限流约束")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--replay", help="不发请求，对该目录下已落盘的事件流重新打分")
    ap.add_argument("--output", help="规则报告另存路径，可在回放时保留原始实测耗时")
    args = ap.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    if args.layer:
        cases = [c for c in cases if c["layer"] in args.layer.split(",")]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    jobs = [(c, v) for c in cases for v in c["versions"] if not args.version or v == args.version]

    headers = {} if args.replay else login(args.base)  # 先登录，失败时不留空目录
    run_dir = args.replay or os.path.join(HERE, "traces", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    trace = lambda c, v: os.path.join(run_dir, f"{c['id']}.{v}.sse")
    if args.replay:
        jobs = [(c, v) for c, v in jobs if os.path.exists(trace(c, v))]
    if not jobs:
        print("没有可跑的用例")
        return 1

    def run(job):
        case, version = job
        fetch_error, started = None, time.time()
        if not args.replay:
            try:
                fetch(args.base, case, version, trace(case, version), args.timeout, headers)
            except Exception as e:  # 服务挂了或超时：记失败，继续跑下一条
                fetch_error = f"{type(e).__name__}: {e}"
        t = parse(trace(case, version), version) if os.path.exists(trace(case, version)) else parse(os.devnull, version)
        if fetch_error:
            t["error"] = t["error"] or fetch_error
        if not args.replay:  # v1 的事件不带时间戳，真跑时以实际计时为准
            t["seconds"] = int(time.time() - started)
        r = score(case, version, t)
        failed = [k for k, v in r["checks"].items() if v is False and k in GATES]
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']} {version} "
              f"{r['seconds'] or '?'}s 报告{r['report_chars']}字 引用{r['refs']}"
              + (f"  不过: {', '.join(failed)}" if failed else "")
              + (f"  缺: {', '.join(r['missing_facts'])}" if r["missing_facts"] else ""), flush=True)
        return r

    with ThreadPoolExecutor(max_workers=1 if args.replay else args.workers) as pool:
        results = list(pool.map(run, jobs))

    metrics = summarize(results)
    ok = all(m["ok"] for m in metrics.values())
    report = {"run_dir": run_dir, "ok": ok, "cases_passed": sum(r["passed"] for r in results),
              "cases_total": len(results), "metrics": metrics, "results": results}
    output = args.output or os.path.join(run_dir, "report.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n用例通过 {report['cases_passed']}/{report['cases_total']}")
    for name, m in metrics.items():
        gate = f"门槛 {m['gate']:.0%}" if m["gate"] is not None else "仅统计"
        rate = format_fraction(m["passed"], m["applicable"])
        print(f"  {'OK  ' if m['ok'] else 'FAIL'} {name:17s} {m['passed']}/{m['applicable']} = {rate}  ({gate})")
    print(f"报告: {output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
