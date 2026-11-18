"""Research-report Markdown and retrieved passages. Offsets always refer to the unmodified strings."""
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from urllib.parse import urlsplit


LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\(((?:https?|local)://(?:[^()\s]|\([^()\s]*\))+)\)')
INDEX = re.compile(r'##(\d+)(?:\$\$|##)')
# Keep the layout and Unicode offsets when masking metadata.
IMAGE = re.compile(r'!\[([^\]\n]*)\]\(([^)\n]*)\)')
META = re.compile(r'\[\^\d+\]|<[^>\n]+>|(?:https?|local)://[^\s)\]）|]+')
CN = '零〇○一二两三四五六七八九十百千万亿'
SCALE = {'万亿': 10**12, '亿': 10**8, '千万': 10**7, '百万': 10**6, '万': 10**4, '千': 1000, '百': 100}
UNITS = (r'个百分点|基点|季度|万元/股|元/股|美元|人民币|港元|欧元|公里|千米|平方米|千克|公斤|'
         r'小时|分钟|个月|年度|年末|月底|周年|亿元|元|股|倍|成|%|％|‰|人|家|名|项|个|'
         r'架次|架|艘|台|件|次|套|笔|单|吨|米|天|日|月|年|季|周|轮|代|型|页|种|条|席|位|'
         r'个百分点|点|秒|GHz|MHz|kHz|Hz|kg|km|bps')
DIGITS = r'(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:[.．]\d+)?'
DECIMAL = rf'(?:{DIGITS}|[.．]\d+)'
CN_VALUE = rf'[零〇○一二两三四五六七八九十][{CN}]*(?:点[{CN}]+)?'
# Match both ends of a range separately. A year or a number in a heading remains a claim.
NUMBER = re.compile(
    rf'(?P<arab>(?:(?<![\d.])[+＋\-−－])?{DECIMAL}(?:[eE][+\-]?\d+)?)'
    rf'(?P<aunit>[ \t]*(?:(?:万亿|千万|百万|亿|万|千)(?:{UNITS})?|{UNITS}))?'
    rf'|(?P<cn>百分之{CN_VALUE}|千分之{CN_VALUE}|{CN_VALUE}分之{CN_VALUE}|一半|半数|第{CN_VALUE}|'
    rf'(?:负)?{CN_VALUE}(?=[ \t]*(?:{UNITS}))|[零〇一二两三四五六七八九十][{CN}]*[十百千万亿])'
    rf'(?P<cunit>[ \t]*(?:{UNITS}))?', re.I)
ORD = rf'(?:\d+(?:\.\d+)*|[{CN}]+)'
PREFIX = re.compile(rf'^\s*(?:>\s*)*(?:#{{1,6}}\s+)?(?:[-*+]\s+)?(?:\*\*)?'
                    rf'(?:第{ORD}(?:章|节|篇|部分|[，、])|[表图]\s*{ORD}(?=[：:\s])|'
                    rf'[（(]{ORD}[）)]|{ORD}[、)）]|{ORD}\.(?!\d))\s*')
HEADING_NUM = re.compile(rf'^\s*(?:#{{1,6}}\s+(?:\*\*)?\d{{1,2}}(?:\.\d+)*|(?:\*\*)?\d{{1,2}}(?:\.\d+)+)'
                         rf'(?:\*\*)?\s+(?!(?:(?:万|亿)?(?:{UNITS})|是|为|到|至))')
HEAD = re.compile(r'^\s*(#{1,6})\s+(.*)')
BIB = re.compile(r'(?:参考(?:文献|资料|来源)|引用(?:文献|来源)|资料来源|数据来源|References|Bibliography)\s*[:：]?\s*$', re.I)
SOURCE_NOTE = re.compile(r'^\s*[>_＊*\-\s]*(?:(?:数据|资料|信息|表格|表中数据)?来源|出处|Source)[*_\s]*[:：]', re.I)
DIVIDER = re.compile(r'^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$')
TABLE_PIPE = re.compile(r'(?<!\\)\|')
EDITORIAL_COUNT = re.compile(rf'(?:本报告|本文|本章|本节|下文|以下)(?:将|只|主要)?(?:回答|讨论|列出|介绍|分为)'
                             rf'\s*(?P<count>(?:\d+|{CN_VALUE})\s*(?:个|项|点|部分)?)'
                             r'(?:具体|事实性?|研究|主要|核心)*(?:问题|方面|要点|步骤|章节)')


def citation_keys(text, version):
    matches = LINK.finditer(text) if version == 'v2' else INDEX.finditer(text)
    return tuple(dict.fromkeys(m[2 if version == 'v2' else 1] for m in matches))


def _blank(m):
    return ' ' * len(m[0])


def _visible(line):
    def link_label(m):
        label = m[1]
        # A linked table value or assertion is still part of the report. Publication titles,
        # years in filenames and numeric reference labels are citation metadata.
        numeric_label = NUMBER.search(label) and (
            re.search(r'营收|收入|利润|增长|下[降滑]|占比|达[到]?|持股|派[息发]|[％%]|\d\s*(?:元|[万亿]元|人|股|架|台)', label)
            or re.fullmatch(rf'{DECIMAL}(?:{UNITS})', label))
        if numeric_label and not re.search(r'\.(?:pdf|docx?|html?)$', label, re.I):
            return ' ' + label + ' ' * (len(m[0])-len(label)-1)
        return _blank(m)

    line = IMAGE.sub(lambda m: '  ' + m[1] + ' ' * (len(m[0])-len(m[1])-2), line)
    line = LINK.sub(link_label, INDEX.sub(_blank, line))
    return META.sub(_blank, line)


def _clean(text):
    clean = ' '.join(_visible(text).replace('**', '').strip().split())
    return re.sub(r'\s+([。！？；，、|])', r'\1', clean)


@dataclass(frozen=True)
class Occurrence:
    start: int
    end: int
    text: str
    context: str
    refs: tuple
    cite_at: int
    line: int
    unit_hint: str = ''

    @property
    def quantity(self):
        if self.text.endswith('个') and '⟧季度' in self.context:
            return self.text[:-1] + '季度'
        return self.text + self.unit_hint

    @property
    def clause(self):
        # Isolate the marked occurrence; the complete context still resolves who, which
        # period and, in a table, which row/column. Keep separators inside grouped digits.
        clauses = re.split(r'[。！？；;|]|(?<!\d)[，,]|[，,](?!\d)', self.context)
        return next((part.strip() for part in clauses if '⟦' in part), self.context)


def _table_scopes(lines, version, ref_urls):
    scopes = {}
    used_notes = set()
    i = 0
    while i + 1 < len(lines):
        if '|' not in lines[i] or not DIVIDER.fullmatch(lines[i + 1]):
            i += 1
            continue
        end = i + 2
        while end < len(lines) and TABLE_PIPE.search(lines[end]) and not DIVIDER.fullmatch(lines[end]):
            if end + 1 < len(lines) and DIVIDER.fullmatch(lines[end + 1]):
                break
            end += 1
        notes = []
        for p, step in ((i - 1, -1), (end, 1)):
            if 0 <= p < len(lines) and not lines[p].strip():
                p += step
            if 0 <= p < len(lines) and p not in used_notes:
                line = lines[p]
                residual = _clean(line).strip(' >*_()（）:：。')
                if (SOURCE_NOTE.match(line) or not residual or
                        (step == -1 and re.match(r'^\s*[*_]*表\s*\d', line))) and citation_keys(line, version):
                    notes.append(line)
                    used_notes.add(p)
        header = lines[i]
        shared = citation_keys(' '.join([header, *notes]), version)
        if not shared:
            unique = citation_keys(' '.join(lines[i:end]), version)
            if len({ref_urls.get(key, key) for key in unique}) == 1:
                shared = unique
        caption = lines[i-1] if i and not HEAD.match(lines[i-1]) and '|' not in lines[i-1] else ''
        if i >= 2 and not caption.strip() and not HEAD.match(lines[i-2]) and '|' not in lines[i-2]:
            caption = lines[i-2]
        for n in [i, *range(i + 2, end)]:
            scopes[n] = (header, shared, _clean(caption))
        i = end
    return scopes


def _cells(line):
    limits = [0, *(m.end() for m in TABLE_PIPE.finditer(line)), len(line) + 1]
    return [(a, b - 1) for a, b in zip(limits, limits[1:])
            if b > a and (a != 0 or line[:b-1].strip()) and (b <= len(line) or line[a:].strip())]


def inventory(report, version, ref_urls=None):
    """Enumerate occurrences, including uncited ones, with local citation and table scope."""
    lines = report.split('\n')
    tables = _table_scopes(lines, version, ref_urls or {})
    claims, headings = [], {}
    offset = 0
    bibliography_level = None
    preceding = ''
    for n, line in enumerate(lines):
        base = offset
        offset += len(line) + 1
        visible = _visible(line)
        for m in EDITORIAL_COUNT.finditer(visible):
            a, b = m.span('count')
            visible = visible[:a] + ' ' * (b-a) + visible[b:]
        heading = HEAD.match(line)
        if heading:
            preceding = ''
            level = len(heading[1])
            if bibliography_level and level <= bibliography_level:
                bibliography_level = None
            if BIB.fullmatch(PREFIX.sub('', heading[2]).strip(' *')):
                bibliography_level = level
            headings = {k: v for k, v in headings.items() if k < level}
        if bibliography_level and (heading or citation_keys(line, version)):
            residual = PREFIX.sub('', LINK.sub('', INDEX.sub('', line)))
            if heading or not re.search(r'增长|下降|营收|利润|占比|派发|持股|\d\s*(?:[%％]|[万亿]?元|人|股|台)', residual):
                continue
        if DIVIDER.fullmatch(line):
            continue
        for pattern in (PREFIX, HEADING_NUM):
            m = pattern.match(line)
            if m:
                visible = ' ' * m.end() + visible[m.end():]
        parent = ' / '.join(headings.values())
        if n in tables:
            header, shared, caption = tables[n]
            cols = _cells(header)
            cells = _cells(line)
            row_keys = citation_keys(line, version)
            for col, (a, b) in enumerate(cells):
                label = _clean(header[slice(*cols[col])]) if col < len(cols) else ''
                if label in ('序号', '#') and re.fullmatch(r'\s*\d+\s*', line[a:b]):
                    continue
                cell = line[a:b]
                keys = citation_keys(cell, version) or row_keys or shared
                context = f'{parent} {caption} 表头 {_clean(header)} 本行 {_clean(line)} 当前列 {label} 单元格'
                row_name = _clean(line[slice(*cells[0])])
                unit = re.search(r'(?:[（(/]|单位[：:\s]*)\s*((?:万亿|千万|百万|亿|万|千)?(?:元/股|亿元|美元|元|股|吨|%|％|人|台|件))',
                                 ' '.join((label, row_name, caption)))
                matches = list(NUMBER.finditer(visible, a, b))
                if not matches:  # A bare Chinese quantity can also occupy a whole table cell.
                    m = re.fullmatch(rf'\s*({CN_VALUE})\s*', visible[a:b])
                    if m:
                        pos, stop = a + m.start(1), a + m.end(1)
                        claims.append(Occurrence(base+pos, base+stop, line[pos:stop], context + ' ⟦' + m[1] + '⟧', keys, base+b, n+1))
                for m in matches:
                    marked = _clean(line[a:m.start()] + '⟦' + m[0] + '⟧' + line[m.end():b])
                    period_cell = label in ('年份', '年度', '报告期', '季度', '期间')
                    hint = unit[1] if unit and m['arab'] and not m['aunit'] and not period_cell else ''
                    if period_cell and re.fullmatch(r'[12]\d{3}', m[0]):
                        hint = '年'
                    claims.append(Occurrence(base+m.start(), base+m.end(), m[0], context + ' ' + marked, keys, base+b, n+1, hint))
            continue
        # Split on prose punctuation only after links have been masked. A citation placed just
        # after a sentence terminator still belongs to that sentence.
        starts = [0]
        for m in re.finditer(r'[。！？!?；;]|(?<!\d)\.(?=\s|$)|(?<=\d)\.(?=\s+[A-Z])', visible):
            p = m.end()
            tail = re.match(r'(?:[ \t]*(?:##\d+(?:\$\$|##)|\[[^\]]+\]\((?:https?|local)://[^)]+\)))+', line[p:])
            starts.append(p + tail.end() if tail else p)
        prior = preceding
        for a, b in zip(starts, starts[1:] + [len(line)]):
            if a >= b:
                continue
            sentence = line[a:b]
            keys = citation_keys(sentence, version)
            cite_at = b
            while cite_at > a and line[cite_at-1] in '。！？!?；; \t':
                cite_at -= 1
            for m in NUMBER.finditer(visible, a, b):
                marked = _clean(line[a:m.start()] + '⟦' + m[0] + '⟧' + line[m.end():b])
                context = ' '.join(p for p in (parent, prior, marked) if p)
                claims.append(Occurrence(base+m.start(), base+m.end(), m[0], context, keys, base+cite_at, n+1))
            prior = _clean(sentence)
        if heading:
            headings[level] = _clean(visible).lstrip('# ').strip()
        elif line.strip() and not SOURCE_NOTE.match(line):
            preceding = _clean(visible)[-200:]
    return claims


def original_text(row):
    """Local summaries are whole verbatim KB chunks; web evidence uses the engine's excerpt."""
    if 'original_text' in row:
        return row['original_text'] or ''
    if row.get('is_local') or str(row.get('url', '')).startswith('local://'):
        return row.get('summary') or row.get('snippet') or ''
    return row.get('snippet') or ''


def source_record(row):
    url, text = row.get('url') or '', original_text(row)
    if not text.strip() or not re.match(r'^(?:https?|local)://', url):
        return None
    try:
        location = urlsplit(url)
    except ValueError:
        return None
    if not location.netloc or (location.path in ('', '/') and not location.query):
        return None
    title = row.get('title') or row.get('name') or row.get('siteName') or ''
    date = row.get('date') or row.get('datePublished') or ''
    chunk = row.get('chunk_index')
    key = '\0'.join((url, str(chunk), title, date, text))
    return dict(source_id=hashlib.sha256(key.encode()).hexdigest()[:24], url=url, title=title,
                date=date, chunk_index=chunk, original_text=text)


def source_catalog(rows):
    catalog = {}
    for row in rows:
        record = source_record(row)
        if record:
            catalog[record['source_id']] = record
    return catalog


def _overlap(before, after):
    return next((n for n in range(min(len(before), len(after), 180), 11, -1)
                 if before.endswith(after[:n])), 0)


def document_context(record, catalog):
    """Identify the document and, for a split local table, its preceding headers in raw retrievals.

    These excerpts can resolve who, when or the unit. The claimed value must still occur in
    the cited passage. Never carry headers between URLs or revisions.
    """
    if not record['url'].startswith('local://'):
        return []
    excerpts = []
    index = record.get('chunk_index')
    for row in catalog.values():
        if (row['url'], row['title'], row['date']) != (record['url'], record['title'], record['date']):
            continue
        if row['source_id'] == record['source_id']:
            continue
        raw = row['original_text']
        # Older traces have the verbatim cover but no chunk_index.
        cover = re.search(r'(?m)^#{1,2}\s+[^\n]{2,100}(?:年度报告|报告摘要)[^\n]*', raw[:220])
        if cover:
            excerpts.append(dict(kind='document_title', source_id=row['source_id'],
                                 quote_start=cover.start(), quote=cover[0]))
        other = row.get('chunk_index')
        if isinstance(index, int) and isinstance(other, int) and abs(index-other) == 1:
            before, after = (raw, record['original_text']) if other < index else (record['original_text'], raw)
            # Local chunks overlap. Only continue a table across an actual matching boundary.
            if '|' in before[-180:] and '|' in after[:180] and _overlap(before, after):
                start = max(0, len(raw)-700) if other < index else 0
                excerpts.append(dict(kind='previous_table_chunk' if other < index else 'next_table_chunk', source_id=row['source_id'],
                                     quote_start=start, quote=raw[start:start+700]))
    return excerpts


def table_excerpt(quote, context):
    """Reassemble only overlapping verbatim excerpts, in document order. This preserves row/column alignment."""
    for part in context:
        other = part['quote']
        if part['kind'] == 'previous_table_chunk' and (n := _overlap(other, quote)):
            quote = other + quote[n:]
        elif part['kind'] == 'next_table_chunk' and (n := _overlap(quote, other)):
            quote += other[n:]
    return quote


@lru_cache(maxsize=4096)
def _decimal(s):
    s = unicodedata.normalize('NFKC', s).replace(',', '').replace('−', '-')
    try:
        return Decimal(s)
    except InvalidOperation:
        pass
    if not s or not re.fullmatch(rf'[{CN}]+(?:点[{CN}]+)?', s):
        return None
    nums = dict(zip('零一二三四五六七八九两〇○', '0123456789200'))
    integer, _, fraction = s.partition('点')
    if all(c in nums for c in integer):
        value = int(''.join(nums[c] for c in integer))
    else:
        total = section = digit = 0
        for char in integer:
            if char in nums:
                digit = int(nums[char])
            elif char in '十百千':
                section += (digit or 1) * {'十': 10, '百': 100, '千': 1000}[char]
                digit = 0
            elif char == '万':
                section = (section + digit) * 10000
                digit = 0
            elif char == '亿':
                total += (section + digit) * 10**8
                section = digit = 0
        value = total + section + digit
    if any(c not in nums for c in fraction):
        return None
    return Decimal(str(value) + ('.' + ''.join(nums[c] for c in fraction) if fraction else ''))


def magnitude(text):
    normalized = unicodedata.normalize('NFKC', text.strip()).replace('−', '-')
    if normalized.startswith(('一半', '半数')):
        return Decimal(50), Decimal(1), '%'
    fraction = re.fullmatch(rf'({CN_VALUE})分之({CN_VALUE})', normalized)
    if fraction:
        denominator = _decimal(fraction[1])
        return (_decimal(fraction[2]) * 100 / denominator, Decimal(1), '%') if denominator else None
    m = re.search(rf'{DECIMAL}(?:[eE][+\-]?\d+)?|{CN_VALUE}', normalized.removeprefix('百分之').removeprefix('千分之'))
    if not m:
        return None
    val = _decimal(m[0])
    if val is None or abs(val.adjusted()) > 80:
        return None
    suffix = normalized[normalized.find(m[0])+len(m[0]):].strip()
    factor = next((v for k, v in SCALE.items() if suffix.startswith(k)), 1)
    # Magnitudes are a candidate filter. The semantic check must verify sign, unit and scope.
    unit = re.sub(r'^(?:万亿|千万|百万|亿|万|千|百)', '', suffix)
    unit = {'年度': '年', '季': '季度', '人民币': '元', '％': '%',
            '名': '人', '位': '人'}.get(unit, unit)
    if normalized.startswith('百分之'):
        unit = '%'
    elif normalized.startswith('千分之'):
        unit = '‰'
    if unit == '成':
        factor, unit = Decimal(factor) * 10, '%'
    elif unit == '‰':
        factor, unit = Decimal(factor) / 10, '%'
    return abs(val), Decimal(factor), unit


@lru_cache(maxsize=4096)
def source_values(text):
    clean = re.sub(r'<br\s*/?>', '', text, flags=re.I)
    measures = {c.start: c.quantity for c in inventory(clean, 'v2') if c.unit_hint} if '|' in clean else {}
    return tuple(v for m in NUMBER.finditer(clean) if (v := magnitude(measures.get(m.start(), m[0]))) is not None)


def contains_value(token, quote):
    target = magnitude(token)
    if not target:
        return False
    value, factor, unit = target
    for candidate, scale, source_unit in source_values(quote):
        if unit and source_unit and unit != source_unit:
            continue
        if candidate == value and (factor == scale or not source_unit or not unit):
            return True
        converted = candidate * scale / factor
        if converted == value:
            return True
        if (factor != scale or value.as_tuple().exponent < 0) and converted and value:
            try:
                if converted.quantize(Decimal(1).scaleb(value.as_tuple().exponent), rounding=ROUND_HALF_UP) == value:
                    return True
            except InvalidOperation:
                continue
    return False

def number_anchor(token, record, pos, quote, catalog):
    """Locate the number itself. A fiscal year can be on a retrieved cover or a split table header."""
    if contains_value(token, quote):
        return dict(source_id=record['source_id'], quote_start=pos, quote=quote)
    parsed = magnitude(token)
    if not parsed or parsed[2] not in ('年', '年末') or not 1900 <= parsed[0] <= 2200:
        return None
    for part in document_context(record, catalog):
        if part['kind'] != 'document_title' and table_excerpt(quote, [part]) == quote:
            continue
        if contains_value(token, part['quote']):
            return {k: part[k] for k in ('source_id', 'quote_start', 'quote')}
    return None


def other_cited_passages(claim, records, primary_ids=()):
    """Other operands of an explicit comparison must come from actual, locally cited passages."""
    if not re.search(r'加总|合计|之和|差额|一致|比重|占全年|相差', claim.context):
        return []
    # Retain companion inputs, never hunt through unrelated documents for the highlighted value.
    operands = set()
    for m in NUMBER.finditer(claim.context.replace('⟦'+claim.text+'⟧', '')):
        q = magnitude(m[0])
        if q and (q[2] in ('元', '元/股', '股', '%') or not q[2] and q[0] >= 10000):
            operands.add(m[0])
    if not operands:
        return []
    matches = []
    for row in records:
        if row['source_id'] in primary_ids:
            continue
        original = row['original_text']
        for pos in range(0, len(original), 720):
            quote = original[pos:pos+960]
            hits = sum(contains_value(v, quote) for v in operands)
            if hits:
                matches.append((hits, row['source_id'], pos, quote))
            if pos + 960 >= len(original):
                break
    return [dict(source_id=sid, quote_start=pos, quote=quote)
            for _, sid, pos, quote in sorted(matches, reverse=True)[:2]]


def clean_missing_relations(report):
    """Drop a comparison/estimate whose result was removed; retain independent numbers and table cells."""
    marker = '〔未核实〕'
    if marker not in report:
        return report, []
    omissions = []
    cleaned = []
    lines = report.split('\n')
    tables = _table_scopes(lines, 'v2', {})
    for n, line in enumerate(lines):
        if marker not in line or n in tables or HEAD.match(line):
            cleaned.append(line)
            continue
        bounds = [0, *(m.end() for m in re.finditer(r'[。！？；;]|(?<!\d)[，,]|[，,](?!\d)', _visible(line))), len(line)]
        parts = []
        for a, b in zip(bounds, bounds[1:]):
            if a == b:
                continue
            part = line[a:b]
            if marker in part:
                left = _visible(part.replace(marker, ''))
                # A verified measure in this same clause must survive. Period labels used only
                # as operands of the missing measure belong to that incomplete comparison.
                qualifiers = ('季度', '年', '年末', '月', '日')
                tokens = [m[0] for m in NUMBER.finditer(left)]
                if all(t.startswith('第') or (magnitude(t) and magnitude(t)[2] in qualifiers) for t in tokens):
                    omissions.append(part.strip())
                    if parts and re.search(r'(?:测算|推算|估算)[，,]$', parts[-1].strip()):
                        omissions.append(parts.pop().strip())
                    continue
            parts.append(part)
        remaining = ''.join(parts).strip()
        if remaining:
            remaining = re.sub(r'^[，,；;]\s*|^其中(?=[一二三四第])', '', remaining)
            if line.rstrip().endswith('。') and remaining.endswith(('，', ',', '；', ';')):
                remaining = remaining[:-1]+'。'
            cleaned.append(remaining)
    return '\n'.join(cleaned).strip() or '现有原文不足以确认所询数值。', omissions
