# 新闻舆情-股价异动关联解释型 RAG

这是一个用于信息系统设计课程展示的 Python 3.11 + FastAPI 原型系统。系统围绕“近期新闻如何与企业、行业和行情异动产生关联”这个问题，自动检索新闻与行情数据，构建多阶段检索流程、图谱增强 RAG、词云、行情总分表和评价面板，最终生成只支持“关联解释”的回答。

系统只做信息分析，不构成投资建议。回答固定为 `claim_level=association_only`，不输出买入、卖出、持有建议，不输出目标价，不做股价预测，不声称确定因果。

## 核心目标

- 将用户自然语言问题转化为企业、行业、时间窗口和新闻检索任务。
- 从真实新闻源与行情源获取近期证据，证据不足时明确说明，不编造。
- 用多阶段检索记录解释“材料是怎样被找到、过滤和进入推理上下文的”。
- 用图谱增强 RAG 展示企业、新闻、主题、来源、行情异动窗口之间的关系。
- 用行情总分表、新闻事件线、词云和评价面板增强答辩展示效果。
- 所有结果均保留风险边界和引用证据。

## WebUI 功能介绍

页面入口为 FastAPI 挂载的 `frontend/index.html`。启动服务后访问：

```bash
http://127.0.0.1:8000
```

页面展示顺序如下：

1. **Answer**：报告式回答，先给总结论，再给关键解释，新闻证据统一放在回答末尾。
2. **可视化数据**：展示行情总分表和新闻词云。
3. **图谱增强 RAG**：将新闻事件线、行情异动窗口和关系图谱结合展示。
4. **企业关系与推理路径**：展示企业间可能的竞争、上下游、生态和风险偏好联动。
5. **多阶段检索流程**：展示实体识别、行情检索、新闻检索、过滤、重排、引用生成等步骤。
6. **行业与企业影响**：展示行业主题命中、关键词和样例企业。
7. **结果可评价**：展示引用数量、新闻数量、行情交易日、数据源、LLM provider、置信度和风险提示。
8. **引用证据**：展示最多 100 条可追溯新闻引用。

## RAG 流程

本项目可以被称为 RAG 原型，因为它具备检索增强生成的关键链路：

1. **Retrieval**：按问题自动识别企业、行业词、时间窗口，检索新闻和行情。
2. **Filtering and Ranking**：结合行情异动日、标题、摘要、来源和主题词对新闻重排。
3. **Augmentation**：把证据、行情摘要、图谱摘要、行业映射和评价指标注入 LLM 上下文。
4. **Generation**：由 LLM 或受控模板生成关联解释。
5. **Guardrail**：风险守卫会清理投资建议、目标价、确定因果等不合规表达。
6. **Evaluation**：输出覆盖率、置信度、风险状态和数据缺失原因。

## 数据源

默认目标是真实数据驱动：

- 新闻：Yahoo Finance 新闻搜索、GDELT 等适配器。
- 行情：Yahoo Finance chart、Stooq 等适配器。
- LLM：DeepSeek OpenAI-compatible Chat Completions。
- 数据库：PostgreSQL，通过 `DATABASE_URL` 连接。

如果外部源失败，系统不会伪造真实数据，而是在 `evaluation.notes`、`missing_real_data_reasons` 和前端风险提示里说明原因。

## DeepSeek API Key

WebUI 支持临时输入 DeepSeek API Key，也支持保存到本地 `.env`。保存后页面只显示不可复制的占位符，可以删除后重新输入。真实密钥不会返回给前端，也不会写入分析结果。

## 本地运行

安装依赖：

```bash
py -3.11 -m pip install -r requirements.txt
```

启动服务：

```bash
py -3.11 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

运行测试：

```bash
py -3.11 -B -m pytest -p no:cacheprovider
```

## Docker 说明

Docker 仅作为可选 PostgreSQL 环境参考，不参与实际开发流程。本项目不会自动创建或启动容器。需要数据库时，可以参考 `docker-compose.example.yml` 自行启动 PostgreSQL，并在 `.env` 中配置 `DATABASE_URL`。

## API

### Health

```http
GET /api/health
```

返回：

```json
{"status":"ok"}
```

### Policy

```http
GET /api/policy
```

返回系统风险边界和 `association_only` 约束。

### Analyze

```http
POST /api/analyze
Content-Type: application/json

{
  "question": "最近 AI 芯片、EV 和利率新闻可能怎样影响各个行业？请举例相关企业，并推理企业之间可能的关系。"
}
```

核心返回字段：

- `answer`
- `claim_level`
- `retrieval_trace`
- `market_summary`
- `news_summary`
- `graph_summary`
- `citations`
- `evaluation`
- `risk_warnings`

## 项目结构

```text
app/
  main.py          FastAPI 路由与静态页面挂载
  rag.py           RAG 主流程、检索、重排、回答组装、持久化
  sources.py       新闻源和行情源适配器
  analytics.py     行情收益率、异常波动、成交量 z-score
  graph.py         图谱增强 RAG 摘要
  insights.py      行业影响、企业关系、词云和推理上下文
  guard.py         风险守卫
  llm.py           DeepSeek / Mock LLM 客户端
  schemas.py       Pydantic 输入输出模型
frontend/
  index.html       无构建工具的单页 WebUI
migrations/
  001_init.sql     PostgreSQL 表结构
diagrams/
  use_case.puml
  class_diagram.puml
  sequence_analyze.puml
tests/
  pytest 测试
```

## 课程图文件

根目录 `diagrams/` 已准备 PlantUML 图：

- `use_case.puml`：UML 用例图，描述参与者、系统边界和主要用例。
- `class_diagram.puml`：UML 类图，展示核心服务、数据源、模型和返回结构。
- `sequence_analyze.puml`：顺序图，描述一次 `/api/analyze` 请求从 WebUI 到 RAG、数据源、LLM、风控和响应的交互过程。

这些文件可直接复制到 PlantUML、VS Code PlantUML 插件或在线渲染器生成图片。

## 风险边界

- 本系统仅用于课程研究和信息分析展示。
- 所有输出均不构成投资建议。
- 新闻与行情的时间重合只能支持关联解释，不能证明新闻导致股价变化。
- 无真实新闻或行情证据时，系统必须明确说明证据不足。
- 外部 API 失败时必须记录失败原因，不得使用假数据冒充真实结果。
