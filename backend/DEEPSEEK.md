# DeepSeek 接入

## 本地配置

在 `backend/.env` 中设置（密钥不写入代码）：

```dotenv
DEEPSEEK_API_KEY=你的官方密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-flash
```

聊天、记忆提取、深度研究 V1/V2、资讯采集使用上述配置。各研究角色仍可通过 `LLM_MODEL_ARCHITECT` 等变量覆盖模型。现有工具循环使用非思考模式，避免丢失推理内容和挤占正文输出预算。

## 联网搜索

统一使用 `POST https://api.deepseek.com/v1/tools/search`，与模型共用密钥。保留标题、原文链接、摘要、正文和发布日期，供现有引用流程使用。限流时最多请求 4 次。

- 搜索词：1–128 个字符；结果数：1–50。
- 旧 `/search/web` 接口保留返回结构；仅支持第一页和 `search` 类型。
- 旧地区、语言、纠错参数仅为客户端兼容保留，不传给官方 API。
- 向量检索与重排继续使用独立的 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL` 配置。

更新配置后重启后端即可生效。

## 验证

已真实验证官方模型回答和搜索来源返回，并通过搜索适配器、研究流程回归及引用防伪检查。未执行完整长篇研究报告生成。

官方文档：[模型接口](https://api-docs.deepseek.com/api/create-chat-completion/)、[搜索接口](https://api-docs.deepseek.com/api/search/)。
