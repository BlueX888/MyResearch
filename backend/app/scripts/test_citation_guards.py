"""引用防伪自检：python app/scripts/test_citation_guards.py"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from service.deep_research_v2.agents.scout import DeepScout

results = [{"url": "https://a.com/news/1.html"}, {"url": "https://b.com/x"}, {"url": "https://b.com/y"}, {"url": "local://kb/k/d"}]
analysis = {"extracted_facts": [
    {"source_url": "https://a.com/news/1.html"},  # 原样保留
    {"source_url": "https://a.com"},              # 首页，同站唯一 -> 还原
    {"source_url": "https://b.com/"},             # 首页，同站两条 -> 清空
    {"source_url": "https://invented.org/z"},     # 自拟 -> 清空
    {"source_url": "local://kb/k/d"},
    {},
]}
DeepScout._pin_fact_sources(analysis, results)
assert [f["source_url"] for f in analysis["extracted_facts"]] == [
    "https://a.com/news/1.html", "https://a.com/news/1.html", "", "", "local://kb/k/d", ""]
DeepScout._pin_fact_sources({}, results)
DeepScout._pin_fact_sources(None, results)

# 与 graph.py 里清理正文链接的正则保持一致
fact_urls = {"https://a.com/news/1.html", "local://kb/k/d"}
strip = lambda t: re.sub(r'(?<!!)\[([^\]]+)\]\(((?:https?|local)://[^)\s]+)\)',
                         lambda m: m.group(0) if m.group(2) in fact_urls else m.group(1), t)
assert strip("据[甲](https://a.com/news/1.html)与[乙](https://fake.com)，见![图](chart_1)和[年报](local://kb/k/d)") == \
    "据[甲](https://a.com/news/1.html)与乙，见![图](chart_1)和[年报](local://kb/k/d)"
print("引用防伪: 全部通过")
