# 最终验证材料

- `scoring-17-tests.log`：当前评测器离线回归17/17。14项日志为扩展前记录。
- `scoring-before-fix-expected-failures.log`：旧脚本负对照，10个失败断言是预期结果。
- `rule-replay-23.*`：18份既有知识库/断供加5份早期研究，23/23；最终21份新解析器重算也通过，见 `latest-parser-replay.json`。
- `judge-selftest-26-of-26.log`：16构造项加10条真实回归，26/26。过程中23/25、25/26日志均保留。不是100条人工盲标校准。
- `independent-contradiction-review.log`：对全量原判的6句C使用生产函数独立复判，4句保留C、2句改P；其他标签与要点/拒答不变。
- `live-validation-4.json`：4个真实HTTP复测全过，原始检索数量与SSE一致、补充原文齐全、审核均生效。
- `backend-fix.log`：实际结构重试、临时网关429、生成图表脚本失败及恢复均保留。
- `manual-review-original-9.json`、`manual-review-retest-1.json`：最初人工判断的历史记录；最终结论以上一级 `manual-review.md/json` 为准，未覆盖历史。
- 新版本地知识库事件分支仅离线回归，未新增本地HTTP实测。

被测系统离线脚本 `app/scripts/test_eval_regressions.py`：5/5。报告内容质量未改。

样本来源、哈希和计时见上一级 `provenance.json`；原始报告/判官和修正结果分文件保存。
