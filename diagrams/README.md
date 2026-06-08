# UML 图说明

本目录保存课程报告使用的 PlantUML 源文件。三张图按作业要求保持简洁。

## 中文译名

- `use_case.puml`：**用例图：系统功能边界、参与者及用例关系**
- `class_diagram.puml`：**类图：核心类及其关系**
- `sequence_analyze.puml`：**顺序图：一次分析请求的对象交互过程**

## 图的用途

- 用例图：描述系统边界、外部参与者和主要功能。
- 类图：选取核心类绘制，体现 API、RAG 编排、数据源、LLM、风险守卫和数据库模型之间的关系。
- 顺序图：选取“用户发起一次分析”作为核心业务场景，展示 WebUI、FastAPI、RAG、数据源、LLM、风险守卫和进度状态之间的交互。

若 `.png` 与 `.puml` 不一致，以 `.puml` 为准。

## 渲染方式

```bash
plantuml diagrams/*.puml
```

也可以使用 VS Code PlantUML 插件或在线 PlantUML 渲染器导出图片。
