# 项目规则

本项目是新闻舆情-股价异动关联解释型 RAG。只做信息分析，不做投资建议。

## 硬性约束

- 不输出买入、卖出、持有建议。
- 不输出目标价。
- 不做股价预测。
- 不声称因果关系。
- 无证据不编造。
- 所有密钥从 `.env` 读取。
- PostgreSQL 通过 `DATABASE_URL` 连接。
- 行情数据来自外部适配器或 mock，不假设本地已有。
- LLM 默认支持 DeepSeek，失败时可降级 mock。
- Embedding 默认本地模型或关闭，不调用云端 embedding。
- 所有文件写入当前仓库。

## 运行命令

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest
```

## 完成标准

- `/api/health` 可用。
- `/api/policy` 可用。
- `/api/analyze` 可返回结构化结果。
- 结果包含 citations。
- 结果包含 risk_warnings。
- 支持至少一个 mock 新闻源和一个 mock 行情源。
- 预留 GDELT、Alpha Vantage、Twelve Data、Nasdaq Data Link、Stooq 适配器。
- 测试通过。
- README 可作为课程报告基础。
