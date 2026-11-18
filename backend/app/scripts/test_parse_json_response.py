"""parse_json_response 自检：python app/scripts/test_parse_json_response.py"""
import os, sys, logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); logging.disable(logging.ERROR)
from service.deep_research_v2.agents.critic import CriticMaster
a = CriticMaster.__new__(CriticMaster); a.name = "t"; a.logger = logging.getLogger("t")
p = a.parse_json_response
assert p('{"a": {"x": 1, "y": 2,}}') == {"a": {"x": 1, "y": 2}}
assert p('```json\n{"a": 1}\n```') == {"a": 1}
assert p('好的，结果如下：\n{"a": 1}') == {"a": 1}
assert p('{"a": 1}\n\n{"b": "c"}') == {"a": 1}                      # 带尾巴
assert p('{"summary": "第一行\n第二行"}') == {"summary": "第一行\n第二行"}  # 未转义换行
# 截断：与探路时审核返回的形状一致，评分和结论必须保住
t = p('{"overall_assessment": {"quality_score": 5, "verdict": "needs_revision", "summary": "有, 逗号"}, '
      '"issues": [{"id": "i1", "severity": "critical", "description": "矛盾"}, {"id": "i2", "sever')
assert t["overall_assessment"] == {"quality_score": 5, "verdict": "needs_revision", "summary": "有, 逗号"}, t
assert t["issues"][0] == {"id": "i1", "severity": "critical", "description": "矛盾"}, t  # 完整的那条保住
assert p('{"s": "截断在字符串里, 后面没了') == {}     # 救不回来就老实返回空
assert p("完全不是 JSON") == {}
print("parse_json_response: 8 项全部通过")
