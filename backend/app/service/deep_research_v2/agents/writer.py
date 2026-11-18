# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - 首席笔杆 Agent (LeadWriter)

职责：
1. 深度写作 - 将零散信息串联成逻辑严密的报告
2. Markdown排版 - 专业的格式排版
3. 图文混排 - 整合文字、图表、数据
4. 参考文献 - 规范的引用格式
"""

import uuid
from typing import Dict, Any, List
from datetime import datetime

from report_numbers import NUMBER, contains_value
from .base import BaseAgent
from ..state import ResearchState, ResearchPhase


NUMBER_RULES = """事实数字只取自下列带链接原文，逐项保持主体、指标、数值、单位、期间及预测/实际口径。
单项事实问题直接回答，表格优先沿用原文的数值、单位和表头。省去题外数据、另算的占比和没有原文的估计。每个数字所在句就近标注原文链接。同源表可在表头或紧邻表格的“数据来源”行共引。
优先写明原文表头、单位和报告期。缺原文时写明信息缺口，不用估算填空。"""


def source_material(state, facts):
    """Preserve the exact quote through section drafting, synthesis and revision."""
    passages = {}
    for f in facts:
        if f.get("source_quote") and f.get("source_id"):
            passages[(f["source_url"], f["source_quote"])] = f.get("source_name") or "原文"
    # Direct retrieval remains available even when the extractor failed to quote a table.
    for r in state.get("raw_sources", []):
        if not r.get("original_text"):
            continue
        passages.setdefault((r["url"], r["original_text"]), r.get("title") or "原文")
    return "\n".join(f"[{name}]({url})\n{quote}" for (url, quote), name in list(passages.items())[:32])


class LeadWriter(BaseAgent):
    """
    首席笔杆 - 最终输出的打磨者

    特点：
    - 深度写作能力
    - 专业的行业研究报告风格
    - 逻辑严密的叙述结构
    - 规范的引用和排版
    """

    SECTION_WRITING_PROMPT = """你是一位顶级投行研究部的首席分析师，擅长撰写深度行业研究报告。

## 研究主题
{query}

## 当前章节信息
标题: {section_title}
描述: {section_description}
类型: {section_type}

## 可用素材

### 相关事实
{facts}

### 数据点
{data_points}

### 已有洞察
{insights}

### 相关图表
{charts_info}

## 写作要求
1. **专业性**：使用行业术语，体现专业深度
2. **逻辑性**：论点清晰，论据充分，层层递进
3. **数据支撑**：关键观点必须有数据或事实支撑
4. **引用规范**：使用可点击链接格式 [来源名称](URL)。URL 必须原样使用事实/来源清单里给出的链接，禁止自拟链接或改写成网站首页；没有链接的内容不要加引用
5. **图表整合**：需要展示的读数用 Markdown 表格写明指标、期间、单位和原文链接。图像供过程页查看。
6. **篇幅**：按材料与问题决定；事实问答直接给出数值与依据
7. **不要重复标题**：正文开头不要再写章节标题

## 输出格式
```json
{{
    "content": "章节正文内容（Markdown格式，不包含章节标题）",
    "key_points": ["本章节的核心要点"],
    "citations": [
        {{"source": "来源名称", "url": "完整URL"}}
    ],
    "suggested_improvements": ["如果有更多信息可以改进的地方"]
}}
```

开始撰写："""

    SYNTHESIS_PROMPT = """你是首席笔杆，需要将各章节整合成完整的研究报告。

## 研究主题
{query}

## 各章节内容
{sections_content}

## 收集的所有引用来源
{all_sources}

## 任务
若用户只问一个具体事实，直接给出答案、时间、口径和原文，可用一张表写全；省去无关章节与其他经营指标。综合研究报告按以下结构整合。
1. 撰写报告摘要（Executive Summary）
2. 整合各章节，确保逻辑连贯，使用层级编号
3. 撰写结论与展望
4. 整理参考文献列表（确保链接可点击）

## 关键要求

### 1. 标题编号规则（必须严格遵守）
- 一级标题：1、2、3...（如：1 市场概况）
- 二级标题：1.1、1.2、2.1...（如：1.1 市场规模）
- 三级标题：1.1.1、1.1.2...（如：1.1.1 全球市场）
- **禁止标题重复**：每个标题必须唯一，不要在正文中重复章节标题

### 2. 引用格式规则（确保可点击）
- 行内引用：使用 [来源名称](URL) 格式。URL 必须原样使用事实/来源清单里给出的链接，禁止自拟链接或改写成网站首页；没有链接的内容不要加引用
- 数据引用：保留章节里的真实原文链接，表格注明单位、期间与来源。
- 文末参考文献：使用有序列表 + 可点击链接格式

### 3. 报告结构规范
- 不要在报告开头使用 # 一级标题
- 直接从"执行摘要"开始
- 各章节使用 ## 二级标题
- 子章节使用 ### 三级标题

## 输出格式
```json
{{
    "executive_summary": "执行摘要（300-500字）",
    "full_report": "完整报告（Markdown格式，按下方结构生成）",
    "conclusions": ["核心结论1", "核心结论2"],
    "outlook": "未来展望",
    "references": [
        {{"id": 1, "title": "来源标题", "url": "完整URL", "author": "作者/机构", "date": "日期"}}
    ]
}}
```

## 报告结构模板
```markdown
## 执行摘要

[300-500字的研究摘要]

---

## 1 [第一章标题]

[章节引言段落]

### 1.1 [子章节标题]

[内容，包含数据引用如：根据[来源名](URL)，...]

### 1.2 [子章节标题]

#### 1.2.1 [三级标题]

[更详细的内容]

---

## 2 [第二章标题]

### 2.1 [子章节标题]

...

---

## 结论与展望

### 核心结论
1. [结论1]
2. [结论2]

### 未来展望
[展望内容]

---

## 参考文献

1. [来源标题1](URL1) - 作者/机构, 日期
2. [来源标题2](URL2) - 作者/机构, 日期
...
```"""

    REVISION_PROMPT = """你是首席笔杆，需要根据审核反馈修订报告。

## 原始报告
{original_content}

## 审核反馈
{feedback}

## 补充的新信息
{new_info}

## 任务
根据反馈修订报告，解决指出的问题。

## 修订原则
1. 针对性修改：只修改有问题的部分
2. 补充来源：对缺少来源的观点补充引用
3. 修正错误：纠正事实错误或逻辑漏洞
4. 保持风格：修订后保持报告整体风格一致

输出JSON：
```json
{{
    "revised_content": "修订后的内容",
    "changes_made": ["修改1", "修改2"],
    "addressed_issues": ["已解决的问题ID"],
    "unable_to_address": ["无法解决的问题及原因"]
}}
```"""

    def __init__(self, llm_api_key: str, llm_base_url: str, model: str = "deepseek-flash"):
        super().__init__(
            name="LeadWriter",
            role="首席笔杆",
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=model
        )

    async def process(self, state: ResearchState) -> ResearchState:
        """处理入口"""
        if state["phase"] == ResearchPhase.WRITING.value:
            return await self._write_report(state)
        elif state["phase"] == ResearchPhase.REVISING.value:
            return await self._revise_report(state)
        else:
            return state

    async def _write_report(self, state: ResearchState) -> ResearchState:
        """撰写报告"""
        # 发送 research_step 开始事件
        # 注意: step_type 必须是 "writing" 以匹配 graph.py 发送的 phase 事件
        self.add_message(state, "research_step", {
            "step_id": f"step_writing_{uuid.uuid4().hex[:8]}",
            "step_type": "writing",
            "title": "内容生成",
            "subtitle": "撰写研究报告",
            "status": "running",
            "stats": {"sections_count": len(state["outline"]), "word_count": 0}
        })

        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "开始撰写深度研究报告..."
        })

        # 逐章节撰写
        for section in state["outline"]:
            if section.get("status") not in ["final", "drafted"]:
                await self._write_section(state, section)

        # 整合报告
        await self._synthesize_report(state)

        # 发送 research_step 完成事件
        word_count = len(state.get("final_report", ""))
        self.add_message(state, "research_step", {
            "step_type": "writing",
            "title": "内容生成",
            "subtitle": "撰写研究报告",
            "status": "completed",
            "stats": {
                "sections_count": len(state["outline"]),
                "word_count": word_count,
                "references_count": len(state.get("references", []))
            }
        })

        # 更新阶段
        state["phase"] = ResearchPhase.REVIEWING.value

        return state

    async def _write_section(self, state: ResearchState, section: Dict) -> None:
        """撰写单个章节"""
        section_id = section["id"]
        self.logger.info(f"Writing section: {section.get('title')}")

        self.add_message(state, "action", {
            "agent": self.name,
            "tool": "writing_section",
            "section": section.get("title")
        })

        # 收集相关素材
        related_facts = [f for f in state["facts"] if section_id in f.get("related_sections", [])]
        if not related_facts:
            # 如果没有特定关联，使用所有事实
            related_facts = state["facts"][:10]

        # 格式化事实
        facts_text = []
        for fact in related_facts:
            # 必须带上真实链接，否则模型只能自己编一个网站首页
            source = f"[{fact.get('source_name')}]({fact['source_url']})" if fact.get("source_url") else f"{fact.get('source_name')}，无链接"
            if NUMBER.search(str(fact.get("content", ""))) and not fact.get("source_quote"):
                continue
            facts_text.append(f"- {fact.get('content')} ({source})")

        facts_text.append(source_material(state, related_facts))

        # 格式化数据点
        data_text = []
        for dp in state["data_points"][:10]:
            quote = dp.get("source_quote") or ""
            if dp.get("source_id") and contains_value(str(dp.get("value", "")) + str(dp.get("unit", "")), quote):
                data_text.append(f"- {dp.get('name')} {dp.get('value')}{dp.get('unit', '')} "
                                 f"{dp.get('year') or ''} [原文]({dp.get('source_url')}) {quote}")

        # 格式化图表信息
        charts_info = []
        for chart in state["charts"]:
            if chart.get("section_id") == section_id:
                charts_info.append(f"- 图表: {chart.get('title')} (ID: {chart.get('id')})")

        prompt = self.SECTION_WRITING_PROMPT.format(
            query=state["query"],
            section_title=section.get("title", ""),
            section_description=section.get("description", ""),
            section_type=section.get("section_type", "mixed"),
            facts="\n".join(facts_text) if facts_text else "（暂无相关事实）",
            data_points="\n".join(data_text) if data_text else "（暂无数据点）",
            insights="\n".join([f"- {i}" for i in state["insights"][:5] if not NUMBER.search(str(i))]) if state["insights"] else "（暂无洞察）",
            charts_info="\n".join(charts_info) if charts_info else "（暂无图表）"
        )

        response = await self.call_llm(
            system_prompt="你是顶级的行业研究分析师，擅长撰写专业的研究报告。",
            user_prompt=prompt + "\n" + NUMBER_RULES,
            json_mode=True,
            temperature=0.4,
            max_tokens=16000  # 拉满到最大值
        )

        result = self.parse_json_response(response)

        if result and result.get("content"):
            section_content = result["content"]
            state["draft_sections"][section_id] = section_content
            section["status"] = "drafted"

            # 收集引用
            for citation in result.get("citations", []):
                state["references"].append({
                    "id": len(state["references"]) + 1,
                    "marker": citation.get("marker"),
                    "source": citation.get("source"),
                    "url": citation.get("url", "")
                })

            # 发送章节内容到"过程报告" - 包含完整内容用于流式显示
            self.add_message(state, "section_content", {
                "agent": self.name,
                "section_id": section_id,
                "section_title": section.get("title"),
                "content": section_content,  # 完整章节内容
                "word_count": len(section_content),
                "key_points": result.get("key_points", [])
            })

            # 发送观察消息（显示在左侧步骤流程）
            self.add_message(state, "observation", {
                "agent": self.name,
                "content": f"章节「{section.get('title')}」撰写完成\n字数: {len(section_content)}\n要点: {', '.join(result.get('key_points', [])[:2]) if result.get('key_points') else '无'}"
            })

    async def _synthesize_report(self, state: ResearchState) -> None:
        """整合完整报告"""
        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "正在整合各章节，生成完整研究报告..."
        })

        # 准备各章节内容
        sections_content = []
        for section in state["outline"]:
            section_id = section["id"]
            content = state["draft_sections"].get(section_id, "")
            if content:
                sections_content.append(f"## {section.get('title')}\n{content}")

        prompt = self.SYNTHESIS_PROMPT.format(
            query=state["query"],
            sections_content="\n\n".join(sections_content) if sections_content else "（暂无章节内容）",
            all_sources=source_material(state, state["facts"])
        )

        self.logger.info(f"[LeadWriter] 调用 LLM 整合报告...")
        response = await self.call_llm(
            system_prompt="你是资深的研究报告主编，擅长整合和打磨最终报告。",
            user_prompt=prompt + "\n" + NUMBER_RULES,
            json_mode=True,
            temperature=0.3,
            max_tokens=16000  # 拉满到最大值
        )

        result = self.parse_json_response(response)
        self.logger.info(f"[LeadWriter] JSON 解析结果: {bool(result)}, keys: {result.keys() if result else 'N/A'}")

        executive_summary = ""
        conclusions = []

        if result and result.get("full_report"):
            state["final_report"] = result.get("full_report", "")
            executive_summary = result.get("executive_summary", "")
            conclusions = result.get("conclusions", [])
            self.logger.info(f"[LeadWriter] ✅ 报告整合成功，长度: {len(state['final_report'])}")

            # 更新参考文献
            for ref in result.get("references", []):
                if ref not in state["references"]:
                    state["references"].append(ref)
        else:
            # JSON 解析失败时的备选方案：使用已有章节内容组装报告
            self.logger.warning(f"[LeadWriter] ⚠️ JSON 解析失败，使用章节内容作为备选")
            fallback_report = f"# {state['query']} 研究报告\n\n"
            for section in state["outline"]:
                section_id = section["id"]
                content = state["draft_sections"].get(section_id, "")
                if content:
                    fallback_report += f"## {section.get('title', section_id)}\n\n{content}\n\n"
            state["final_report"] = fallback_report
            self.logger.info(f"[LeadWriter] 使用备选报告，长度: {len(state['final_report'])}")

        # 发送报告完成事件 - 包含完整报告内容用于前端流式显示
        self.add_message(state, "report_draft", {
            "agent": self.name,
            "content": state["final_report"],  # 完整报告内容
            "executive_summary": executive_summary,
            "conclusions": conclusions,
            "word_count": len(state["final_report"]),
            "references_count": len(state["references"])
        })

    async def _revise_report(self, state: ResearchState) -> ResearchState:
        """根据反馈修订报告"""
        self.add_message(state, "thought", {
            "agent": self.name,
            "content": "根据审核反馈修订报告..."
        })

        # 收集未解决的问题
        unresolved = [f for f in state["critic_feedback"] if not f.get("resolved")]
        feedback_text = []
        for issue in unresolved:
            feedback_text.append(f"- [{issue.get('severity')}] {issue.get('description')}\n  建议: {issue.get('suggestion')}")

        # 收集新信息（如果有补充搜索）
        new_facts = state["facts"][-5:] if state["facts"] else []
        new_info = source_material(state, new_facts)

        prompt = self.REVISION_PROMPT.format(
            original_content=state.get("final_report", ""),
            feedback="\n".join(feedback_text) if feedback_text else "无具体反馈",
            new_info=new_info if new_info else "无补充信息"
        )

        response = await self.call_llm(
            system_prompt="你是负责修订报告的资深编辑。",
            user_prompt=prompt + "\n" + NUMBER_RULES,
            json_mode=True,
            temperature=0.3,
            max_tokens=16000  # 拉满到最大值
        )

        result = self.parse_json_response(response)

        if result and result.get("revised_content"):
            state["final_report"] = result["revised_content"]

            # 标记已解决的问题
            for issue_id in result.get("addressed_issues", []):
                for feedback in state["critic_feedback"]:
                    if feedback.get("id") == issue_id:
                        feedback["resolved"] = True

            self.add_message(state, "revision_complete", {
                "agent": self.name,
                "changes_count": len(result.get("changes_made", [])),
                "addressed_issues": result.get("addressed_issues", []),
                "unable_to_address": result.get("unable_to_address", [])
            })

        # 回到审核阶段
        state["phase"] = ResearchPhase.REVIEWING.value

        return state
