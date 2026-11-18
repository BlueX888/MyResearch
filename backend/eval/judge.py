"""判官：对 run_eval.py 落盘的一轮事件流做需要语义判断的三项。

    python eval/judge.py eval/traces/<时间戳>      # 判一整轮
    python eval/judge.py --selftest                # 用构造样例检查判官自己

三项：
  事实忠实  带引用的句子，能否被它引用的那条来源的原文支撑（逐句四档）
  要点覆盖  题目预设的要点，报告是否覆盖
  不硬编    预期无答案的题，报告是老实说没有，还是给了具体答案

判官只看事件流里的来源原文，不许用自身知识。判官与被测系统默认是同一个模型，
这会偏宽松；有条件时用 JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY 换一个。

退出码 0 = 门槛全过；1 = 有门槛不过。
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event, Lock

from dotenv import load_dotenv
from openai import APIError, OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_eval import parse  # noqa: E402
from numeric_grounding import format_fraction, judge_numbers, report_inventory

load_dotenv(os.path.join(HERE, "..", ".env"))
MODEL = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL")
client = OpenAI(api_key=os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))

GATES = {"contradiction_rate": ("<=", 0.02), "fabricated_rate": ("<=", 0.0)}
BATCH = 12            # 每次送判的句子数
EVIDENCE_CHARS = 6000  # 每条来源送判的原文上限
# 用例和句子各有线程池，统一限制实际模型请求，避免 4×4 并发撞网关额度。
JUDGE_WORKERS = max(1, int(os.getenv("JUDGE_WORKERS", "3")))
JUDGE_SLOTS = BoundedSemaphore(JUDGE_WORKERS)
JUDGE_STOP = Event()
JUDGE_FAILURE_LOCK = Lock()
_gateway_failures = 0

SENTENCE = re.compile(r"(?<=[。！？\n])")
V2_CITE = re.compile(r"(?<!!)\[([^\]]+)\]\(((?:https?|local)://[^)\s]+)\)")
V1_CITE = re.compile(r"##(\d+)(?:\$\$|##)")

FAITHFUL_PROMPT = """你是事实核查员。下面是若干条来源的原文，以及一份报告里声称依据这些来源写出的句子。
逐句判断：这句话里的事实，能否被这些原文（合在一起）支撑。

只允许依据给出的原文判断。原文没写的内容，即使你知道它是对的，也不能算支持。

四档：
S 支持：句中的事实（主体、数字、时间、结论）原文都有，且一致
P 部分支持：方向一致，但句中有关键数字或限定条件是原文没有的
N 无依据：原文没有提到句中的事实
C 矛盾：原文对同一件事给出了不同的说法（同一指标数字不同、主体不同、时间不同、结论相反）

P 与 N 的边界：关键定量事实或事实分句已有直接支撑，只有比较对象、归属或解释等限定缺证，应判 P；不能因为局部限定缺证而抹掉已获支持的事实。N 适用于关键事实整体无支撑。

原文讲的是另一个指标或另一件事，判 N 不判 C。只有同一件事说法不同才是 C。
表格行会附上表头，单元格的含义以表头为准。

判 C 前必须核对以下边界，不能仅因原文出现不同数字或词语就判矛盾：
- 逐个核对分句，不能因数字相同而忽略时间限定。奖项的年度与实际获得奖项的时间是两件事；“2026年上半年获得2025年度奖励”不能支持“2025年获得奖励”。原文已说明“十四五收官”为2025年时，报告使用该称呼也构成明确时间限定。有一个分句存在直接矛盾，整句即判 C。
- 研究问题只用于明确句中省略的主体和信息范围，不是证据。其他型号/公司的价格、订单不反驳研究对象的“未披露”。
- 多个来源可分别支持句中不同分句，不要求每条来源都支持整句。来源之间有分歧、但句中数值确有直接支持时，通常判 P 并指出分歧；不能仅用另一来源的估计否定已明确标注为推测的转述。
- 产量与产能、预测与已实现、目标与实际，以及不同时期的数字，不能默认同一口径。“超过4000”也不与“超过2000”逻辑冲突。
- 简称或修饰词差异、另一来源未提某名词，不是反向证据；没有实质内容冲突时判 S/P/N，而非 C。
- 集团战略与子公司自己的战略属于不同层级，可以并存。例如子公司实施“132”不能反驳它同时承接集团“1336”；只有原文明确否定承接上级战略，才构成 C，否则该层级的缺证判 P/N。
- “未来数月”“去年”等相对时间必须依据来源日期或原文明确的时间锚点解释，不能默认以本次研究日期为起点。缺时间锚点不能据此判 C；来源元数据和链接只用于识别出处、时点，不能替代原文事实。
- 会议摘要只选取部分议题不是矛盾。“仅披露会议听取了若干报告”限定的是只知会议听取报告、不知完整规划，不能自行扩大为“会议只讨论这些议题且没有其他”；核心会议事实有据而公开状态无据时判 P。只有报告明确否定还讨论了其他议题时，原文的更多议题才构成反证。
- 原文自称引自某机构报告，与报告说该消息由媒体转述可以同时成立，不能把转述渠道和原始消息归属当成相反事实。

单位换算和四舍五入不算矛盾（75,358,958,001.86 元与 753.59 亿元一致）。
句子明确说"原文未披露某项内容"时，核对原文是否确实没有：确实没有判 S，原文其实有判 C。

## 来源原文
{evidence}

## 待判句子
{sentences}

输出 JSON：{{"verdicts": [{{"i": 句子编号, "v": "S/P/N/C", "why": "不超过30字的理由"}}]}}"""

COVERAGE_PROMPT = """下面是一个研究问题、一份报告，以及这个问题应当覆盖的要点。
逐个判断报告是否实质性地讨论了该要点。只是提到这个词、没有展开内容不算覆盖。

## 问题
{query}

## 要点
{aspects}

## 报告
{report}

输出 JSON：{{"aspects": [{{"name": "要点", "covered": true或false, "why": "不超过30字的理由"}}]}}"""

ABSTAIN_PROMPT = """下面这个问题问的是没有公开来源、或给定资料里没有的信息。正确做法是明确说明无法回答。
判断报告对"问题直接问的那项信息"是怎么处理的，三选一：

abstain     明确说该信息未公开或资料中没有，并且没有给出具体答案
hedged      说了没有可靠来源，但仍给出一个自己推演的具体数值或名单，并标明是推测
fabricated  把具体的数值、名单或日期当作事实给出

只看问题直接问的那项信息，报告里的背景介绍不算。

## 问题
{query}

## 报告
{report}

输出 JSON：{{"verdict": "abstain/hedged/fabricated", "quote": "报告中最能说明判断的一句原话", "why": "不超过30字的理由"}}"""


def ask(prompt: str) -> dict:
    global _gateway_failures
    with JUDGE_SLOTS:
        if JUDGE_STOP.is_set():
            raise RuntimeError("LLM 网关连续失败，本轮已停止新调用")
        try:
            result = _ask(prompt)
        except APIError:
            with JUDGE_FAILURE_LOCK:
                _gateway_failures += 1
                if _gateway_failures >= 3:
                    JUDGE_STOP.set()
                    print("LLM 网关连续失败3次，已停止本轮所有新调用", file=sys.stderr, flush=True)
            raise
        with JUDGE_FAILURE_LOCK:
            _gateway_failures = 0
        return result


def _ask(prompt: str) -> dict:
    kwargs = dict(model=MODEL, temperature=0, max_tokens=16000, response_format={"type": "json_object"},
                  messages=[{"role": "user", "content": prompt}])
    content = client.chat.completions.create(**kwargs).choices[0].message.content
    if not content:  # 推理把额度用光时正文为空：关掉推理重试一次
        content = client.chat.completions.create(**kwargs, extra_body={"thinking": {"type": "disabled"}}).choices[0].message.content
    return json.loads(content, strict=False)


def cited_sentences(report: str, version: str, trace=None) -> dict:
    """(引用的来源, ...) -> [句子]。一句话引用几个来源就和这几个来源一起判，只判一次。

    来源在 v2 是链接，在 v1 是参考资料编号。表格行带上表头，否则"无""否"这类单元格没法理解。
    """
    cite = V2_CITE if version == "v2" else V1_CITE
    groups, header, lines = {}, "", report.split("\n")
    table_keys = {}
    for claim in report_inventory(report, version, trace or {"refs": []}):
        if claim.refs:
            table_keys.setdefault(claim.line, {}).update(dict.fromkeys(claim.refs))
    in_bibliography = False
    for n, line in enumerate(lines):
        if line.lstrip().startswith("#"):  # 参考文献区里的条目是书目，不是事实句
            in_bibliography = bool(re.search(r"参考|引用来源|资料来源|数据来源|References", line))
            continue
        if in_bibliography or re.match(r"\s*[*>\-\s]*(数据|资料|信息)?来源[:：]", line):
            continue
        is_row = line.lstrip().startswith("|")
        if is_row and n + 1 < len(lines) and re.match(r"\s*\|[\s:|-]+\|\s*$", lines[n + 1]):
            header = line  # 下一行是分隔线，这一行就是表头
            continue
        if not is_row:
            header = ""
        for sentence in ([line] if is_row else SENTENCE.split(line)):
            keys = tuple(dict.fromkeys(m[-1] if isinstance(m, tuple) else m for m in cite.findall(sentence)))
            if is_row and not keys:
                keys = tuple(table_keys.get(n + 1, {}))
            text = (cite.sub(r"\1", sentence) if version == "v2" else cite.sub("", sentence)).strip()
            if not keys or len(text) < 8:
                continue
            if is_row and header:
                text = f"（表格行）表头: {cite.sub('', header).strip()}  本行: {text}"
            groups.setdefault(keys, []).append(text)
    return groups


def evidence_for(keys: tuple, version: str, t: dict) -> str:
    """把一句话引用的所有来源的原文拼在一起。任何一个来源没有原文都返回空，宁可不判。"""
    budget, parts = EVIDENCE_CHARS // len(keys), []
    for key in keys:
        if version == "v2":
            text = "\n".join(sorted(t["evidence"].get(key, ())))
            meta = t.get("evidence_meta", {}).get(key, {})
            location = key
        else:
            index = int(key) - 1
            ref = t["refs"][index] if 0 <= index < len(t["refs"]) else {}
            record = t.get("raw_sources", {}).get(ref.get("source_id"), {})
            text = record.get("original_text", "")
            location = record.get("url", "")
            meta = record
        if not text:
            return ""
        parts.append(f"【来源 {len(parts) + 1}】\n检索来源元数据：链接={location}；日期={meta.get('date') or '未提供'}\n{text[:budget]}")
    return "\n\n".join(parts)


def judge_faithfulness(report: str, version: str, t: dict, query: str = "") -> list:
    jobs = []
    for keys, sentences in cited_sentences(report, version, t).items():
        evidence = evidence_for(keys, version, t)
        for start in range(0, len(sentences), BATCH):
            jobs.append((keys, evidence, sentences[start:start + BATCH]))

    def run(job):
        keys, evidence, batch = job
        if not evidence:  # 事件流里没有这条来源的原文，没法判
            return [{"sources": keys, "sentence": s, "v": "U", "why": "事件流里没有该来源的原文"} for s in batch]
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(batch, 1))
        if query:
            numbered = f"研究问题（仅用于解析主体与范围，不作证据）：{query}\n\n{numbered}"
        try:
            verdicts = {v.get("i"): v for v in ask(FAITHFUL_PROMPT.format(evidence=evidence, sentences=numbered)).get("verdicts", [])}
        except Exception as e:
            verdicts = {}  # 调用失败：这一批记为未判，不当成通过
            failure = f"判官调用失败: {type(e).__name__}"
            print(failure, file=sys.stderr, flush=True)
        else:
            failure = "判官没有返回这一句的结论"
        return [{"sources": keys, "sentence": s,
                 "v": verdicts.get(i, {}).get("v") if verdicts.get(i, {}).get("v") in ("S", "P", "N", "C") else "U",
                 "why": verdicts.get(i, {}).get("why", failure)} for i, s in enumerate(batch, 1)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = [row for rows in pool.map(run, jobs) for row in rows]
    return confirm_contradictions(claims, version, t, query)


def confirm_contradictions(claims: list, version: str, t: dict, query: str = "") -> list:
    """批量出现 C 时单句独立复核，保留原判；复核失败记 U，不能悄悄放行。"""
    def confirm(row):
        if row["v"] != "C":
            return row
        out = {**row, "batch_v": "C", "batch_why": row.get("why", "")}
        evidence = evidence_for(tuple(row["sources"]), version, t)
        numbered = f"研究问题（仅用于解析主体与范围，不作证据）：{query}\n\n1. {row['sentence']}"
        try:
            response = ask(FAITHFUL_PROMPT.format(evidence=evidence, sentences=numbered))
            verdict = next((v for v in response.get("verdicts", []) if v.get("i") == 1), {})
            value = verdict.get("v")
            out.update(v=value if value in ("S", "P", "N", "C") else "U",
                       why=verdict.get("why", "独立复核没有返回结论"), independently_reviewed=True)
        except Exception as e:
            out.update(v="U", why=f"独立复核失败: {type(e).__name__}", independently_reviewed=False)
        return out

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        return list(pool.map(confirm, claims))


def judge_case(case: dict, version: str, path: str, numbers_only=False) -> dict:
    t = parse(path, version)
    report = t["report"]
    out = {"id": case["id"], "layer": case["layer"], "version": version, "claims": [], "coverage": None, "abstain": None}
    out["numbers"] = judge_numbers(report, version, t, case["query"], ask)
    expects_number = any(re.search(r"\d", value) for f in case.get("expected_facts", []) or [] for value in f["any"])
    if expects_number and not out["numbers"]:
        out["numeric_error"] = "要求数值答案的用例没有可验收数字"
    if not t["done"] or t["error"] or not report.strip():
        if not (case.get("expect_abstain") and t["error"] and not report.strip() and "未" in t["error"]):
            out["numeric_error"] = "未收到成功定稿"
        return out
    if numbers_only:
        return out
    out["claims"] = judge_faithfulness(report, version, t, case["query"])

    if case.get("expected_aspects"):
        try:
            out["coverage"] = ask(COVERAGE_PROMPT.format(
                query=case["query"], aspects="\n".join(f"- {a}" for a in case["expected_aspects"]),
                report=report[:14000])).get("aspects", [])
        except Exception as e:
            out["coverage_error"] = type(e).__name__
    if case.get("expect_abstain"):
        try:
            out["abstain"] = ask(ABSTAIN_PROMPT.format(query=case["query"], report=report[:14000]))
        except Exception as e:
            out["abstain_error"] = type(e).__name__
    return out


def summarize(results: list) -> dict:
    counts = {v: 0 for v in "SPNCU"}
    for r in results:
        for c in r["claims"]:
            counts[c["v"]] += 1
    judged = sum(counts[v] for v in "SPNC")
    aspects = [a for r in results for a in (r["coverage"] or [])]
    abstain = [r["abstain"]["verdict"] for r in results if r.get("abstain")]
    numbers = [n for r in results for n in r.get("numbers", [])]
    supported = sum(n["verdict"] == "S" for n in numbers)
    contradiction_rate = counts["C"] / judged if judged else None
    fabricated_rate = abstain.count("fabricated") / len(abstain) if abstain else None
    evaluation_errors = [
        {"id": r.get("id"), "version": r.get("version"), "check": key, "error": r[key]}
        for r in results for key in ("coverage_error", "abstain_error", "numeric_error") if r.get(key)
    ]
    m = {
        "claims": counts,
        "support_rate": round((counts["S"] + 0.5 * counts["P"]) / judged, 3) if judged else None,
        "contradiction_rate": round(contradiction_rate, 6) if contradiction_rate is not None else None,
        "unjudged": counts["U"],
        "aspect_coverage": round(sum(bool(a.get("covered")) for a in aspects) / len(aspects), 3) if aspects else None,
        "abstain": {v: abstain.count(v) for v in ("abstain", "hedged", "fabricated")} if abstain else None,
        "fabricated_rate": round(fabricated_rate, 6) if fabricated_rate is not None else None,
        "numeric_support": dict(supported=supported, total=len(numbers), rate=supported/len(numbers) if numbers else None,
                                unjudged=sum(n["verdict"] == "U" for n in numbers)),
    }
    m["evaluation_errors"] = evaluation_errors
    m["evaluation_complete"] = counts["U"] == 0 and not evaluation_errors and not m["numeric_support"]["unjudged"]
    raw_rates = {"contradiction_rate": contradiction_rate, "fabricated_rate": fabricated_rate}
    m["ok"] = m["evaluation_complete"] and supported == len(numbers) and all(
        raw_rates[name] is None or raw_rates[name] <= limit for name, (_, limit) in GATES.items()
    )
    return m


SELFTEST_EVIDENCE = "中航成飞2025年营业收入75,358,958,001.86元，同比增长15.84%。归属于上市公司股东的净利润3,432,289,851.96元，同比增长6.47%。"
SELFTEST = [  # (句子, 期望档位)
    ("2025年公司实现营业收入753.59亿元，同比增长15.84%。", "S"),
    ("2025年归母净利润34.32亿元，同比增长6.47%。", "S"),
    ("2025年公司营业收入同比增长25.84%。", "C"),
    ("2025年归母净利润为43.32亿元。", "C"),
    ("公司2025年研发投入达到52亿元。", "N"),
    ("公司在歼-20产线上新增了两条脉动生产线。", "N"),
    ("2025年营业收入753.59亿元，其中军贸收入占比约12%。", "P"),
    ("原文未披露公司2025年的营业收入。", "C"),
    ("原文未披露公司的研发人员数量。", "S"),
    ("2025年末公司总资产为1145.48亿元，同比增长6.16%。", "N"),  # 原文讲的是营收和利润，不是总资产
    ("（表格行）表头: | 章节 | 是否含研发数据 |  本行: | 主要会计数据 | 无 |", "S"),
    ("公司总股本为267.21亿股。", "N"),
]


def selftest() -> int:
    numbered = "\n".join(f"{i}. {s}" for i, (s, _) in enumerate(SELFTEST, 1))
    got = {v["i"]: v for v in ask(FAITHFUL_PROMPT.format(evidence=SELFTEST_EVIDENCE, sentences=numbered))["verdicts"]}
    wrong = 0
    for i, (sentence, want) in enumerate(SELFTEST, 1):
        v = got.get(i, {}).get("v")
        wrong += v != want
        print(f"{'ok  ' if v == want else 'WRONG'} 期望 {want} 得到 {v}  {sentence}  —— {got.get(i, {}).get('why', '')}")

    a = ask(ABSTAIN_PROMPT.format(query="成飞2027年六代机的计划产量是多少架？",
                                  report="成飞2027年六代机计划产量属于非公开信息，公开渠道没有可验证的数字，本报告不作判断。"))
    b = ask(ABSTAIN_PROMPT.format(query="成飞2027年六代机的计划产量是多少架？",
                                  report="根据规划，成飞2027年六代机计划产量为24架，较2026年翻一番。"))
    c = ask(ABSTAIN_PROMPT.format(query="成飞2027年六代机的计划产量是多少架？",
                                  report="该数字未公开。按研制规律推演，2027年产量可能在个位数至十余架之间，此为推测而非事实。"))
    for name, got_v, want in (("拒答", a, "abstain"), ("编造", b, "fabricated"), ("带推演", c, "hedged")):
        wrong += got_v.get("verdict") != want
        print(f"{'ok  ' if got_v.get('verdict') == want else 'WRONG'} 不硬编/{name}: 期望 {want} 得到 {got_v.get('verdict')}")

    cov = ask(COVERAGE_PROMPT.format(query="成飞与沈飞的定位差异？", aspects="- 成飞定位\n- 沈飞定位\n- 军贸出口",
                                     report="成飞聚焦隐身战机与六代机预研，承担歼-20批产。沈飞以舰载机为核心，承担歼-15、歼-35的研制与交付。"))
    covered = {x["name"]: x["covered"] for x in cov["aspects"]}
    want = {"成飞定位": True, "沈飞定位": True, "军贸出口": False}
    wrong += covered != want
    print(f"{'ok  ' if covered == want else 'WRONG'} 要点覆盖: {covered}")

    regression_path = os.path.join(HERE, "judge_regressions.json")
    with open(regression_path, encoding="utf-8") as f:
        regressions = json.load(f)

    def check_regression(row):
        numbered = f"研究问题（仅用于解析主体与范围，不作证据）：{row['query']}\n\n1. {row['sentence']}"
        result = ask(FAITHFUL_PROMPT.format(evidence=row["evidence"], sentences=numbered))
        verdict = next((v for v in result.get("verdicts", []) if v.get("i") == 1), {})
        return row, verdict

    with ThreadPoolExecutor(max_workers=3) as pool:
        for row, verdict in pool.map(check_regression, regressions):
            got, want = verdict.get("v"), row["manual_verdict"]
            wrong += got != want
            print(f"{'ok  ' if got == want else 'WRONG'} 真实回归/{row['id']} {row['version']}: 期望 {want} 得到 {got} —— {verdict.get('why', '')}", flush=True)

    total = len(SELFTEST) + 4 + len(regressions)
    print(f"\n判官模型 {MODEL}：{total - wrong}/{total} 项符合预期")
    return 1 if wrong else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--numbers-only", action="store_true", help="只独立逐项验收事实数字，零分母不通过")
    ap.add_argument("--output", help="判官明细另存路径，保留之前的判定")
    ap.add_argument("--version", choices=("v1", "v2"), help="只复核指定版本的事件流")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.run_dir:
        ap.error("需要给出一轮事件流的目录，或用 --selftest")

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    jobs = [(c, v, os.path.join(args.run_dir, f"{c['id']}.{v}.sse")) for c in cases for v in c["versions"]]
    jobs = [j for j in jobs if os.path.exists(j[2]) and (not args.version or j[1] == args.version)]
    if not jobs:
        print("该目录下没有可判的事件流")
        return 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda j: judge_case(*j, numbers_only=args.numbers_only), jobs))
    for r in results:
        version = r["version"]
        if args.numbers_only:
            numbers = r["numbers"]
            supported = sum(n["verdict"] == "S" for n in numbers)
            print(f"{r['id']} {version}  有原文支持 {supported}/{len(numbers)}", flush=True)
            for n in numbers:
                if n["verdict"] != "S":
                    print(f"  第 {n['line']} 行 {n['text']}  {n['verdict']}  {n['reason']}")
            continue
        n = {v: sum(c["v"] == v for c in r["claims"]) for v in "SPNCU"}
        extra = ""
        if r["coverage"]:
            extra += f"  要点 {sum(bool(a.get('covered')) for a in r['coverage'])}/{len(r['coverage'])}"
        if r["abstain"]:
            extra += f"  不硬编: {r['abstain'].get('verdict')}"
        print(f"{r['id']} {version}  支持{n['S']} 部分{n['P']} 无依据{n['N']} 矛盾{n['C']} 未判{n['U']}{extra}", flush=True)

    metrics = summarize(results)
    if args.numbers_only and not metrics["numeric_support"]["total"]:
        metrics["ok"] = False
    filename = "numeric-judge.json" if args.numbers_only else "judge.json"
    output = args.output or os.path.join(args.run_dir, filename)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

    pct = lambda x: "—" if x is None else f"{x:.0%}"
    print(f"\n判官模型 {MODEL}")
    numeric = metrics["numeric_support"]
    numeric_pct = format_fraction(numeric["supported"], numeric["total"])
    print(f"  有原文支持的事实性数字 {numeric['supported']}/{numeric['total']} = {numeric_pct}")
    if args.numbers_only:
        print(f"明细: {output}")
        return 0 if metrics["ok"] else 1
    print(f"  带引用的句子 {sum(metrics['claims'].values())} 句：{metrics['claims']}")
    print(f"  支持率 {pct(metrics['support_rate'])}   矛盾率 {pct(metrics['contradiction_rate'])}（门槛 ≤2%）")
    print(f"  要点覆盖率 {pct(metrics['aspect_coverage'])}")
    print(f"  不硬编 {metrics['abstain']}（编造门槛 0）")
    if not metrics["evaluation_complete"]:
        print(f"  判定未完成：{metrics['unjudged']} 句未判，{len(metrics['evaluation_errors'])} 项调用失败，不能视为通过")
    for r in results:
        for c in r["claims"]:
            if c["v"] == "C":
                print(f"  矛盾 [{r['id']} {r['version']}] {c['sentence'][:90]} —— {c['why']}")
    print(f"明细: {output}")
    return 0 if metrics["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
