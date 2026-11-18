"""Regression cases for the numeric denominator, independent raw evidence, repair and final SSE."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from numeric_grounding import check_provenance, judge_numbers
from report_numbers import clean_missing_relations, contains_value, document_context, inventory, number_anchor, other_cited_passages, table_excerpt, source_catalog, source_record
from run_eval import parse, score, summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from service.numeric_review import finalize_numbers, ReviewUnavailable


def source(url, body, chunk=None):
    return dict(url=url, title='年度统计', original_text=body, chunk_index=chunk)


class NumericMarkdownTests(unittest.TestCase):
    def test_count_occurrences_in_headings_lists_and_multiple_claims(self):
        for version, ref in [('v1', '##7$$'), ('v2', '[年报2025](https://src.example/2025/12)')]:
            report = (f'## 1.2 营收2025年\n1. 营收为-1,210.3万元，下降5.2% {ref}。\n'
                      f'（2）预计范围3–5万元。\n### 第三章 分析\n一、共二十家。\n'
                      f'## 参考文献\n1. {ref} 2025-10-30')
            numbers = inventory(report, version)
            self.assertEqual([c.text for c in numbers], ['2025年', '-1,210.3万元', '5.2%', '3', '5万元', '二十家'])
            self.assertTrue(all(report[c.start:c.end] == c.text for c in numbers))

    def test_same_number_in_two_clauses_is_not_one_verdict(self):
        numbers = inventory('甲营收10元，乙净利润10元。', 'v1')
        self.assertEqual(len(numbers), 2)
        self.assertNotEqual(numbers[0].context, numbers[1].context)
        self.assertIn('乙净利润⟦10元⟧', numbers[1].context)

    def test_table_shares_a_citation_within_bounds_and_keeps_cell_units(self):
        for version, ref, key in [('v1', '##2##', '2'), ('v2', '[原文](https://example.org/r)', 'https://example.org/r')]:
            report = (f'|序号|报告期|收入（亿元）|\n|---|---|---|\n|1|2025年|33.09|\n'
                      f'|2|2024年|31|\n\n**数据来源**：{ref}\n\n'
                      '|项目|人数|\n|---|---|\n|员工|36|\n\n收入增加3元')
            numbers = inventory(report, version)
            self.assertEqual([c.text for c in numbers], ['2025年','33.09','2024年','31','36','3元'])
            self.assertTrue(all(c.refs == (key,) for c in numbers[:4]))
            self.assertEqual(numbers[-1].refs, ())
            self.assertEqual(numbers[-2].refs, ())
            self.assertEqual(numbers[1].quantity, '33.09亿元')
            self.assertTrue(contains_value(numbers[1].quantity, '|营业收入|3,308,640,285.35|'))

    def test_mixed_source_table_never_unions_unrelated_rows(self):
        numbers = inventory('|公司|金额|\n|---|---|\n|甲|12元 ##1$$|\n|乙|15元 ##2$$|\n|丙|18元|', 'v1')
        self.assertEqual([c.refs for c in numbers], [('1',), ('2',), ()])

    def test_two_chunks_of_one_document_can_be_shared_with_an_uncited_table_row(self):
        report = '|指标|金额|\n|---|---|\n|甲|12元 ##1$$|\n|乙|15元 ##2$$|\n|丙|18元|'
        numbers = inventory(report, 'v1', {'1':'local://kb/annual', '2':'local://kb/annual'})
        self.assertEqual(numbers[-1].refs, ('1', '2'))

    def test_plain_subheadings_and_headings_starting_with_real_quantities(self):
        report = '2.1 经营分析\n## 12 亿元投资\n## 2025 年财务情况'
        self.assertEqual([c.text for c in inventory(report, 'v2')], ['12 亿元', '2025 年'])

    def test_editorial_question_count_does_not_hide_business_counts(self):
        report = '本报告回答一个具体事实问题：甲公司2025年新开两家门店。'
        self.assertEqual([c.text for c in inventory(report, 'v2')], ['2025年', '两家'])

    def test_citation_after_punctuation_and_numeric_link_label(self):
        numbers = inventory('收入[增长15%](https://example.org/2020)。 次年收入减少2元。 ##1$$', 'v1')
        self.assertEqual([c.text for c in numbers], ['15%', '2元'])
        self.assertEqual(numbers[1].refs, ('1',))

    def test_whole_values_dimensions_and_rounding(self):
        self.assertFalse(contains_value('2.3%', '12.3%'))
        self.assertFalse(contains_value('3亿元', '3万元'))
        self.assertFalse(contains_value('3万元', '3万股'))
        self.assertTrue(contains_value('15.8%', '同比增长15.84%'))
        self.assertFalse(contains_value('15.9%', '同比增长15.84%'))
        self.assertTrue(contains_value('753.59亿元', '本期营业收入75,358,958,001.86元'))
        self.assertTrue(contains_value('1亿元', '单位：万元\n|项目|本期|\n|---|---|\n|营收|10,000|'))
        self.assertTrue(contains_value('二十家', '共有20家'))
        self.assertEqual([c.text for c in inventory('其中三分之一为出口，半数为内销。', 'v2')], ['三分之一','半数'])
        self.assertTrue(contains_value('30%', '三成'))
        self.assertTrue(contains_value('三分之一', '三分之一'))
        self.assertTrue(contains_value('一半', '50%'))
        self.assertFalse(contains_value('2025年', '|2025年末|2024年末|'))
        self.assertFalse(contains_value('2025年末', '|2025年|2024年|'))

    def test_table_context_requires_same_document_and_matching_chunk_boundary(self):
        records = source_catalog([
            source('local://kb/a', '# 甲公司2025年年度报告摘要', 0),
            source('local://kb/a', '|年度|2025年末|合并口径和会计数据|', 1),
            source('local://kb/a', '|2025年末|合并口径和会计数据|\n|净资产|250元|', 2),
            source('local://kb/a', '|无关表|乙公司销售额|', 3),
            source('local://kb/b', '# 乙公司2025年年度报告摘要', 0),
        ])
        row = list(records.values())[2]
        context = document_context(row, records)
        self.assertEqual([e['kind'] for e in context], ['document_title', 'previous_table_chunk'])
        self.assertTrue(all(e['quote'] in records[e['source_id']]['original_text'] for e in context))
        self.assertEqual(table_excerpt(row['original_text'],context), '|年度|2025年末|合并口径和会计数据|\n|净资产|250元|')

    def test_reporting_year_can_use_real_cover_and_body_from_the_same_document(self):
        records = source_catalog([source('local://kb/a', '# 甲公司2025年年度报告摘要', 0),
                                  source('local://kb/a', '分季度营业收入。第一季度10元。', 5),
                                  source('local://kb/b', '# 乙公司2024年年度报告摘要', 0)])
        cover, row, _ = records.values()
        anchor = number_anchor('2025年', row, 0, row['original_text'], records)
        self.assertEqual(anchor['source_id'], cover['source_id'])
        self.assertEqual(anchor['quote'], cover['original_text'])
        self.assertIsNone(number_anchor('2024年', row, 0, row['original_text'], records))
        self.assertIsNone(number_anchor('2025元', row, 0, row['original_text'], records))

    def test_removed_ratio_leaves_the_independently_supported_comparison(self):
        report = '第三季度和第四季度合计占全年收入约〔未核实〕，其中第三季度略高于第四季度 ##1$$。'
        cleaned, omitted = clean_missing_relations(report)
        self.assertEqual(cleaned, '第三季度略高于第四季度 ##1$$。')
        self.assertEqual(len(omitted), 1)
        original = '|季度|占比|\n|---|---|\n|第三季度|〔未核实〕|'
        self.assertEqual(clean_missing_relations(original), (original, []))

    def test_total_can_use_both_explicit_citations(self):
        first = source_record(source('local://kb/a', '|第一季度|第二季度|\n|10元|20元|', 1))
        annual = source_record(source('local://kb/a', '甲全年营业收入30元。', 2))
        report = '全年收入30元 ##2$$，两个季度加总与全年一致 ##1$$。'
        claim = inventory(report, 'v1')[1]
        self.assertEqual(claim.refs, ('2', '1'))
        peers = other_cited_passages(claim, [first, annual], [first['source_id']])
        self.assertEqual(peers[0]['quote'], annual['original_text'])


class FinalizationTests(unittest.IsolatedAsyncioTestCase):
    async def decide(self, system, prompt):
        data = json.loads(prompt)
        items = []
        for job in data['items']:
            p = job['candidates'][0]
            items.append(dict(i=job['i'], p=p, subject=True, value=True, unit=True, time=True))
        return json.dumps(dict(items=items))

    async def test_lookup_before_removal_and_reaudit_of_final_report(self):
        for version in ('v1', 'v2'):
            rows = [source('local://kb/document', '甲公司利润10元。', 0)]
            new = source('local://kb/document', '乙公司利润20元。', 1)
            lookups = []

            async def search(q):
                lookups.append(q)
                return [new] if '20元' in q else []

            report, audit = await finalize_numbers('甲公司利润10元。\n乙公司利润20元。\n丙公司利润90元。',
                                                   version, '各公司利润', rows, self.decide, search)
            self.assertTrue(any('20元' in q for q in lookups))
            self.assertTrue(any('90元' in q for q in lookups))
            self.assertNotIn('90', report)
            self.assertEqual([r['text'] for r in audit['removed']], ['90元'])
            self.assertEqual(audit['supported'], 2)
            refs = [dict(link=r['url'], source_id=source_record(r)['source_id']) for r in rows]
            t = dict(raw_sources=source_catalog(rows), refs=refs, numeric_audit=audit)
            claims = inventory(report, version)
            self.assertEqual(check_provenance(claims, version, t)['supported'], 2)
            audit['items'][1]['quote'] = '我是模型事后编写的原文20元'
            self.assertEqual(check_provenance(claims, version, t)['supported'], 1)

    async def test_uncertain_semantics_for_same_literal_not_accepted(self):
        rows = [source('local://kb/doc', '甲公司2025年净利润20元。')]
        answer = json.dumps({'items':[dict(i=1, p=1, subject=False, value=True, unit=True, time=True)]})
        search = AsyncMock(return_value=[])
        report, audit = await finalize_numbers('乙公司的净利润20元。', 'v2', '乙公司', rows,
                                               AsyncMock(return_value=answer), search)
        search.assert_awaited_once()
        self.assertNotIn('20', report)
        self.assertIsNone(audit['rate'])

    async def test_verifier_failure_does_not_publish_a_vacuous_success(self):
        rows = [source('local://kb/doc', '甲营收12万元。')]
        with self.assertRaises(ReviewUnavailable):
            await finalize_numbers('甲营收12万元。', 'v1', '甲营收', rows, AsyncMock(return_value='{}'),
                                   AsyncMock(return_value=[]))

    async def test_correct_table_footnote_is_preserved(self):
        rows = [source('local://kb/doc', '甲公司营收100元，净利20元。')]
        report = '|指标|值|\n|---|---|\n|营收|100元|\n|净利|20元|\n\n来源：##1$$'
        result, audit = await finalize_numbers(report, 'v1', '甲公司', rows, self.decide, AsyncMock())
        self.assertEqual(result, report)
        self.assertEqual(audit['supported'], 2)

    async def test_selected_passage_must_stand_on_its_own(self):
        rows = [source('https://example.org/a', '乙公司利润20元。'),
                source('https://example.org/b', '丙公司利润20元。')]
        async def judge(system, prompt):
            data = json.loads(prompt)
            pick = data['items'][0]['candidates'][0] if len(data['passages']) > 1 else None
            return json.dumps(dict(items=[dict(i=1,p=pick,subject=True,value=True,unit=True,time=True)]))
        lookup = AsyncMock(return_value=[])
        report, audit = await finalize_numbers('甲公司利润20元。', 'v2', '甲公司', rows, judge, lookup)
        lookup.assert_awaited_once()
        self.assertNotIn('20', report)
        self.assertEqual(audit['total'], 0)


class IndependentMetricTests(unittest.TestCase):
    def test_web_provider_summary_cannot_prove_a_number_absent_from_the_excerpt(self):
        url = 'https://example.org/results'
        hit = dict(url=url, snippet='甲公司营收15元。', summary='甲公司营收53元。')
        events = [dict(type='search_result_item', result=hit), dict(type='reference_materials', content=[hit])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'web.sse'
            path.write_text(''.join('data: '+json.dumps(e)+'\n' for e in events))
            t = parse(path, 'v1')
        claim = judge_numbers('甲公司营收53元 ##1$$', 'v1', t, '', lambda _: self.fail('搜索摘要被当成原文'))
        self.assertEqual(claim[0]['verdict'], 'N')

    def test_legacy_v1_cannot_use_another_chunk_at_same_url(self):
        url = 'local://kb/annual'
        events = [dict(type='search_result_item', result=dict(url=url, summary=s)) for s in ('甲营收15元。', '乙营收30元。')]
        events.append(dict(type='reference_materials', content=[dict(url=url, summary='甲营收15元。')]))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'case.sse'
            p.write_text(''.join('data: '+json.dumps(e)+'\n' for e in events))
            t = parse(p, 'v1')
        rows = judge_numbers('甲营收30元 ##1$$', 'v1', t, '甲', lambda _: self.fail('跨切片送判'))
        self.assertEqual([r['verdict'] for r in rows], ['N'])

    def test_uncited_and_partially_supported_sentences_enter_denominator(self):
        row = source('https://example.org/report', '甲营收10元。')
        trace = dict(raw_sources=source_catalog([row]), refs=[dict(link=row['url'])],
                     numeric_audit={'total':0, 'items':[]})
        report = '甲营收10元、利润80元 [原文](https://example.org/report)。另有19人。'
        def ask(prompt):
            data = json.loads(prompt.split('\n')[-1])
            j = data['claims'][0]
            self.assertEqual(len(data['claims']), 1)
            return dict(items=[dict(i=j['i'], p=j['passages'][0], verdict='S', reason='原文逐字支持')])
        self.assertEqual([r['verdict'] for r in judge_numbers(report, 'v2', trace, '', ask)], ['S','N','N'])

    def test_zero_denominator_does_not_certify_a_numeric_answer(self):
        trace = dict(report='暂无资料', refs=[], retrieved=[], done=True, error=None, review=None,
                     unreviewed=False, iterations=0, seconds=1)
        case = dict(id='probe', layer='kb_fact', search_modes=['local'], expected_facts=[dict(name='金额', any=['12'])])
        result = score(case, 'v1', trace)
        self.assertFalse(result['checks']['numeric_provenance'])
        self.assertIsNone(summarize([result])['numeric_provenance']['rate'])

    def test_year_in_reference_title_does_not_satisfy_required_report_period(self):
        url = 'https://example.org/2025/annual'
        trace = dict(report=f'收入10元 [甲公司2025年年报]({url})。', refs=[dict(link=url,content='原文')],
                     retrieved=[url], done=True, error=None, review=None, unreviewed=False, iterations=0, seconds=1)
        case = dict(id='year', layer='kb_fact', search_modes=['web'],
                    expected_facts=[dict(name='报告期间', any=['2025年'], original_value='2025年')])
        self.assertEqual(score(case, 'v2', trace)['missing_facts'], ['报告期间'])


if __name__ == '__main__':
    unittest.main()
