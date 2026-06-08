import json
from typing import Any

import httpx

from .config import Settings
from .guard import DISCLAIMER


class LLMClient:
    async def generate(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        market_summary: dict[str, Any],
        news_summary: dict[str, Any],
        graph_summary: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    async def generate(self, question, evidence, market_summary, news_summary, graph_summary):
        if not evidence:
            return {
                "answer": f"当前证据不足，无法给出可靠的关联解释。\n\n{DISCLAIMER}",
                "provider": "mock",
                "warnings": ["没有可引用新闻证据，已返回证据不足。"],
            }
        lines = [
            "基于当前可引用证据，新闻事件与行情异动在时间上存在重合，可能存在关联迹象。",
            f"问题：{question}",
            f"行情数据覆盖 {market_summary.get('market_data_days', 0)} 个交易日，识别到 {len(market_summary.get('abnormal_moves', []))} 个主要异动日。",
            f"检索到 {news_summary.get('news_count', 0)} 条新闻，其中 {news_summary.get('event_window_news_count', 0)} 条位于异动窗口附近。",
            "这些材料只能支持关联解释，不能证明新闻必然造成股价变化。",
            "",
            DISCLAIMER,
        ]
        return {"answer": "\n".join(lines), "provider": "mock", "warnings": []}


class DeepSeekLLMClient(LLMClient):
    def __init__(self, settings: Settings, api_key_override: str | None = None):
        self.settings = settings
        self.api_key_override = api_key_override

    async def generate(self, question, evidence, market_summary, news_summary, graph_summary):
        api_key = _clean_api_key(self.api_key_override or self.settings.deepseek_api_key)
        if not evidence:
            return {
                "answer": f"当前证据不足，无法给出可靠的关联解释。\n\n{DISCLAIMER}",
                "provider": "deepseek_not_called",
                "warnings": ["没有真实新闻引用证据，因此未调用 DeepSeek。"],
            }
        if not api_key or api_key == "replace-if-needed":
            return {
                "answer": f"已完成真实行情和新闻证据检索，但 DeepSeek API key 未配置，无法生成 LLM 解释。\n\n{DISCLAIMER}",
                "provider": "deepseek_missing_key",
                "warnings": ["DeepSeek API key 未配置；真实数据已检索，但未使用 mock 生成替代答案。"],
            }

        payload = {
            "model": self.settings.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是课程项目中的金融信息分析助手。只做新闻与行情的关联解释，"
                        "不得给出投资建议、目标价、交易信号或因果断言。答案必须基于证据。"
                    ),
                },
                {
                    "role": "system",
                    "content": (
                        "请用中文写成接近期刊论文摘要与讨论部分的报告正文，不要只列提纲。"
                        "推荐结构为：总结论、研究材料、关键发现、机制解释、行业与企业影响、局限性。"
                        "每个关键判断至少用一段文字解释其逻辑链条；新闻证据可以用角标引用，但不要在正文中重复逐条罗列证据清单。"
                        "必须点名若干具体新闻标题或标题中的事实线索，说明它们为什么映射到相应行业、企业关系或行情窗口。"
                        "可以给出定性情景推演，例如后续应观察哪些变量和行业传导路径，但不得给出股价预测、目标价或交易建议。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "evidence": evidence,
                            "market_summary": market_summary,
                            "news_summary": news_summary,
                            "graph_summary": graph_summary,
                            "required_disclaimer": DISCLAIMER,
                            "writing_requirements": [
                                "输出一篇有连续段落的中文总结报告，风格接近期刊论文摘要和讨论部分。",
                                "先给总结论，再解释关键机制，最后讨论局限性。",
                                "不要输出投资建议、目标价、买卖信号或确定性因果断言。",
                                "不要把新闻证据逐条重复成清单，证据列表由系统在回答末尾另附。",
                                "必须结合至少 6 条具体新闻标题或标题事实线索展开解释。",
                                "加入定性情景推演：说明若新闻主题持续或反转，可能影响哪些行业叙事、企业关系和风险偏好。",
                                "如果证据不足，请明确说明不足之处，但仍要解释现有证据能够支持的关联层级。",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 3000,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.deepseek_request_timeout_seconds,
                proxy=self.settings.proxy_url or None,
            ) as client:
                resp = await client.post(
                    f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer, "provider": "deepseek", "warnings": []}
        except Exception as exc:
            return {
                "answer": f"已完成真实行情和新闻证据检索，但 DeepSeek 调用失败，无法生成 LLM 解释。\n错误摘要：{exc}\n\n{DISCLAIMER}",
                "provider": "deepseek_failed",
                "warnings": [f"DeepSeek 调用失败；真实数据已检索，但未使用 mock 生成替代答案：{exc}"],
            }


def get_llm_client(settings: Settings, api_key_override: str | None = None) -> LLMClient:
    if settings.use_mock_data or settings.llm_provider == "mock":
        return MockLLMClient()
    return DeepSeekLLMClient(settings, api_key_override)


def _clean_api_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    return api_key.strip().strip('"').strip("'")
