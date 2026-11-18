"""Black-box checks on final Markdown and retrieval SSEs. No producer verdict is input to the judge."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
from report_numbers import contains_value, document_context, inventory, number_anchor, other_cited_passages, table_excerpt


def format_fraction(supported, total):
    if not total:
        return '不适用'
    if supported == total:
        return '100%'
    return f'{supported/total:.2%}'.replace('100.00%', '<100%')


def report_inventory(report, version, trace):
    ref_urls = {str(i): r["link"] for i, r in enumerate(trace["refs"], 1)} if version == "v1" else {}
    return inventory(report, version, ref_urls)


def eligible(claim, version, trace):
    raw = trace.get('raw_sources', {})
    keys = set(claim.refs)
    if version == 'v1':
        ids = {r.get('source_id') for i, r in enumerate(trace['refs'], 1) if str(i) in keys}
        return [r for sid, r in raw.items() if sid in ids]
    urls = keys & {r['link'] for r in trace['refs']}
    return [r for r in raw.values() if r['url'] in urls]


def check_provenance(claims, version, trace):
    """Independently enumerate, resolve the in-scope citation, and replay raw string offsets."""
    audit = trace.get('numeric_audit')
    rows = audit.get('items') if isinstance(audit, dict) else []
    rows = rows if isinstance(rows, list) else []
    indexed = {}
    for row in rows:
        if isinstance(row, dict):
            indexed.setdefault((row.get('start'), row.get('end')), []).append(row)
    supported, missing = 0, []
    for c in claims:
        candidates = indexed.get((c.start, c.end), [])
        ok = False
        if len(candidates) == 1:
            item = candidates[0]
            sid, quote, pos = item.get('source_id'), item.get('quote'), item.get('quote_start')
            cited = {r['source_id']:r for r in eligible(c, version, trace)}
            peers = item.get('other_cited_passages', [])
            valid_peers = isinstance(peers, list) and all(
                isinstance(e, dict) and isinstance(e.get('source_id'), str) and e['source_id'] in cited
                and type(e.get('quote_start')) is int and e['quote_start'] >= 0 and isinstance(e.get('quote'), str)
                and bool(e['quote']) and cited[e['source_id']]['original_text'].startswith(e['quote'], e['quote_start'])
                for e in peers)
            for record in cited.values():
                context = item.get('document_context', [])
                expected = {(e['source_id'], e['quote_start'], e['quote'])
                            for e in document_context(record, trace['raw_sources'])}
                valid_context = isinstance(context, list) and all(
                    isinstance(e, dict) and isinstance(e.get('source_id'), str) and type(e.get('quote_start')) is int
                    and isinstance(e.get('quote'), str) and (e['source_id'], e['quote_start'], e['quote']) in expected
                    for e in context)
                if (record['source_id'] == sid and record['url'] == item.get('url') and c.text == item.get('text')
                        and valid_context and valid_peers and isinstance(quote, str) and quote and type(pos) is int and pos >= 0
                        and record['original_text'][pos:pos+len(quote)] == quote
                        and ((item.get('value_anchor') is None and contains_value(c.quantity, quote)) or
                             (item.get('value_anchor') and item['value_anchor'] == number_anchor(c.quantity, record, pos, quote, trace['raw_sources'])))
                        and isinstance(item.get('checks'), dict)
                        and all((item.get('checks') or {}).get(k) is True for k in ('subject', 'value', 'unit', 'time'))):
                    ok = True
                    break
        if ok:
            supported += 1
        else:
            missing.append(dict(line=c.line, start=c.start, text=c.text, context=c.context))
    return dict(supported=supported, total=len(claims), rate=supported/len(claims) if claims else None,
                missing=missing, extra_items=len(rows)-len(claims) if len(rows)>len(claims) else 0)


PROMPT = """请独立复核最终报告，检索片段是唯一的证据。每项的 ⟦ ⟧ 精确标出本次待判数字。
先单独写出 clause 中高亮值的原子断言，再用 context 补回真正限定这个值的主体、指标、数值方向、单位及期间。
只对这个原子断言给 S/N。同一句其他分句、其他数字、推算结果、编辑解释和“未核实”标记分别处理。
例如原文“每箱100件”，报告“按每⟦100件⟧装一箱估算，运费尚缺依据”：高亮值描述装箱标准，应判 S。
反例：原文写甲每箱100件，而报告将该标准归给乙，或改成每批100件，则判 N。
检查表头行列及相邻期间，不得用同一材料中的另一个主体、指标、时期或不同股本口径凑数。
区别实际、预案、预测；年份只示文件名不能证明该年发生某事件。允许正确的等值换算和四舍五入。
附带的同文档标题可辨认主体；table 是同一文档逐字重叠的连续表格，可由行、列及表头联合定位，通常数字出自 original 的行或表头，报告期年份可由同文档原始封面和该期间的具体指标表共同证明。允许从原表直接比较高低、排序和清点明确列出的季度。报告重排行列、另设“数据来源”栏和自拟小标题都不构成缺证，判的是高亮数字的事实含义。研究问题只用于消歧。primary 从本项 passages 选择。对于加总或比较，other_citations 是本句实际引用的其他原文，可用以核对另一个操作数。不能用常识或其他项的片段。
返回 JSON {"items":[{"i":1,"verdict":"S 或 N","p":片段编号或null,"fact":"只写高亮数字的原子断言","reason":"简述其主体、值、单位与时间如何对应或缺失"}]}。
S 表示该数字的完整断言有原文直接支持，部分支持、矛盾及证据不足均为 N。"""


def judge_numbers(report, version, trace, question, ask):
    claims = report_inventory(report, version, trace)
    outcomes, waiting = [], []
    for i, claim in enumerate(claims):
        result = dict(start=claim.start, end=claim.end, line=claim.line, text=claim.text, context=claim.context,
                      verdict='N', reason='引用范围内没有含此数值的检索原文')
        outcomes.append(result)
        options = []
        terms = set(claim.context)
        for record in eligible(claim, version, trace):
            original = record['original_text']
            for pos in range(0, len(original), 800):
                quote = original[pos:pos+1100]
                if number_anchor(claim.quantity, record, pos, quote, trace['raw_sources']):
                    options.append((len(terms.intersection(quote)), record['source_id'], pos, quote))
                if pos + 1100 >= len(original):
                    break
        if options:
            waiting.append((i, sorted(options, reverse=True)[:8]))
    # Batch comparable passages; a verdict can only choose evidence assigned to this occurrence.
    for start in range(0, len(waiting), 10):
        batch = waiting[start:start+10]
        passages, jobs, allowed = {}, [], defaultdict(dict)
        for i, options in batch:
            for _, sid, pos, quote in options:
                row = trace['raw_sources'][sid]
                p = f'{sid}@{pos}'
                context = document_context(row, trace['raw_sources'])
                combined = table_excerpt(quote, context)
                passages[p] = dict(title=row['title'], date=row['date'], original=quote,
                                   document_titles=[v['quote'] for v in context if v['kind']=='document_title'])
                if combined != quote:
                    passages[p]['table'] = combined
                allowed[i][p] = (row['url'], sid, pos, quote)
            companions = other_cited_passages(claims[i], eligible(claims[i], version, trace),
                                              [options[0][1]] if len(options) == 1 else ())
            jobs.append(dict(i=i, clause=claims[i].clause, context=claims[i].context, number=claims[i].text,
                             other_citations=[dict(title=trace['raw_sources'][e['source_id']]['title'], original=e['quote']) for e in companions],
                             passages=list(allowed[i])))
        data = json.dumps(dict(question=question, claims=jobs, passages=passages), ensure_ascii=False)
        try:
            response = ask(PROMPT + '\n' + data)
            answers = response['items']
            if (not isinstance(answers, list) or len(answers) != len(jobs)
                    or {a['i'] for a in answers} != set(allowed)):
                raise ValueError('判官遗漏或重复数字')
            for a in answers:
                i = a['i']
                outcomes[i]['fact'] = a.get('fact', '')
                if a.get('verdict') == 'S' and a.get('p') in allowed[i]:
                    url, sid, pos, quote = allowed[i][a['p']]
                    outcomes[i].update(verdict='S', reason=a.get('reason'), url=url, source_id=sid,
                                       quote_start=pos, quote=quote,
                                       other_cited_passages=other_cited_passages(claims[i], eligible(claims[i], version, trace), [sid]),
                                       value_anchor=number_anchor(claims[i].quantity, trace['raw_sources'][sid], pos, quote, trace['raw_sources']),
                                       document_context=document_context(trace['raw_sources'][sid], trace['raw_sources']))
                elif a.get('verdict') == 'N':
                    outcomes[i].update(reason=a.get('reason'))
                else:
                    outcomes[i].update(verdict='U', reason='判官给出无效证据或结论')
        except Exception as exc:
            for i, _ in batch:
                outcomes[i].update(verdict='U', reason=f'核对未完成 {type(exc).__name__}')
    return outcomes
