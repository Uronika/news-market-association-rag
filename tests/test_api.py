from datetime import date
import time

import anyio
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import get_db
from app.llm import DeepSeekLLMClient
from app.main import app
from app.rag import run_analysis
from app.rag import _clean_markdown_artifacts
from app.schemas import AnalyzeRequest, AnalyzeResponse, CompanySearchResult


def _test_settings() -> Settings:
    settings = Settings()
    settings.use_mock_data = True
    settings.news_provider = "mock"
    settings.market_provider = "mock"
    settings.llm_provider = "mock"
    settings.embed_provider = "none"
    return settings


def _no_db():
    yield None


app.dependency_overrides[get_settings] = _test_settings
app.dependency_overrides[get_db] = _no_db
client = TestClient(app)


def _minimal_analysis_response(*, partial: bool = False) -> AnalyzeResponse:
    return AnalyzeResponse(
        answer="progress test answer",
        claim_level="association_only",
        retrieval_trace=[],
        market_summary={"market_data_days": 0, "abnormal_moves": []},
        news_summary={"news_count": 0},
        graph_summary={"nodes": 0, "edges": 0, "nodes_data": [], "links": []},
        citations=[],
        evaluation={"partial_result": partial, "llm_provider": "test"},
        risk_warnings=["test warning"],
    )


def _wait_for_job(job_id: str) -> dict:
    data = {}
    for _ in range(30):
        response = client.get(f"/api/analyze/progress/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] in {"completed", "partial", "failed"}:
            return data
        time.sleep(0.05)
    return data


def test_health_returns_ok():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_policy_returns_boundaries():
    response = client.get("/api/policy")
    data = response.json()

    assert response.status_code == 200
    assert data["purpose"] == "association analysis only"
    assert data["not_investment_advice"] is True
    assert data["no_causal_claim"] is True


def test_analyze_without_manual_dates_returns_structured_result():
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "最近新闻与 TSLA 近期股价波动之间有什么可能的关联？",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["claim_level"] == "association_only"
    assert data["citations"]
    assert data["risk_warnings"]
    assert data["evaluation"]["llm_provider"] == "mock"
    assert data["evaluation"]["real_data_mode"] is False
    assert data["evaluation"]["data_coverage"]["requested_start_date"]
    assert data["evaluation"]["data_coverage"]["requested_end_date"]
    assert "deepseek_api_key" not in response.text


def test_analyze_start_exposes_request_level_progress(monkeypatch):
    async def fake_run_analysis(request, settings, db, progress=None):
        if progress:
            progress({"stage": "新闻检索", "percent": 46, "message": "新闻检索完成。", "detail": {"news_rows": 7}})
        return _minimal_analysis_response()

    monkeypatch.setattr("app.main.run_analysis", fake_run_analysis)
    response = client.post("/api/analyze/start", json={"question": "progress test"})

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    progress = _wait_for_job(job_id)
    result = client.get(f"/api/analyze/result/{job_id}").json()

    assert progress["status"] == "completed"
    assert progress["percent"] == 100
    assert result["ready"] is True
    assert result["result"]["evaluation"]["partial_result"] is False


def test_analyze_start_marks_partial_result(monkeypatch):
    async def fake_run_analysis(request, settings, db, progress=None):
        if progress:
            progress({"stage": "部分结果", "percent": 100, "message": "达到时间上限，已返回部分结果。"})
        return _minimal_analysis_response(partial=True)

    monkeypatch.setattr("app.main.run_analysis", fake_run_analysis)
    response = client.post("/api/analyze/start", json={"question": "partial test"})

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    progress = _wait_for_job(job_id)
    result = client.get(f"/api/analyze/result/{job_id}").json()

    assert progress["status"] == "partial"
    assert result["ready"] is True
    assert result["result"]["evaluation"]["partial_result"] is True


def test_analyze_returns_retrieval_trace_and_graph_visual_data():
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "最近一个月 TSLA 为什么波动？",
        },
    )
    data = response.json()

    stages = [item["stage"] for item in data["retrieval_trace"]]
    graph_summary = data["graph_summary"]

    assert "问题解析与参数推断" in stages
    assert "实体与问题关键词识别" in stages
    assert any("行情" in stage or "琛" in stage for stage in stages)
    assert graph_summary["nodes"] > 0
    assert graph_summary["edges"] > 0
    assert graph_summary["nodes_data"]
    assert graph_summary["links"]
    assert {"Company", "Ticker", "NewsArticle", "EventWindow", "MarketMove"} <= {
        node["type"] for node in graph_summary["nodes_data"]
    }


def test_evaluation_explains_coverage_and_confidence():
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "最近一个月 TSLA 为什么波动？",
        },
    )
    evaluation = response.json()["evaluation"]

    assert evaluation["coverage_status"]
    assert evaluation["risk_status"]
    assert evaluation["confidence_reasons"]
    assert evaluation["data_coverage"]["market_days"] >= 0


def test_analysis_supports_word_cloud_and_industry_reasoning():
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "最近 AI chip、EV 和 interest rate 新闻可能怎样影响各个行业？",
        },
    )
    data = response.json()
    context = data["news_summary"]["reasoning_context"]

    assert response.status_code == 200
    assert data["evaluation"]["data_coverage"]["minimum_news_candidates"] == 100
    assert data["evaluation"]["data_coverage"]["llm_evidence_limit"] == 50
    assert data["evaluation"]["reasoning_mode"] == "exploratory_hypothesis"
    assert context["word_cloud"]
    assert context["industry_impacts"]
    assert "## 主体分析" in data["answer"]
    assert "## 证据强度与局限" in data["answer"]


def test_analyze_auto_selects_entity_from_question_without_company_inputs():
    response = client.post(
        "/api/analyze",
        json={
            "question": "Tesla 和 NVIDIA 最近的 AI 与 EV 新闻有什么行业影响？",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["news_summary"]["reasoning_context"]["primary_entity"]["ticker"] == "TSLA"
    assert data["graph_summary"]["annotations"]


def test_llm_key_status_does_not_expose_secret(monkeypatch):
    monkeypatch.setattr("app.main.has_deepseek_api_key", lambda: True)
    response = client.get("/api/llm-key")
    data = response.json()

    assert response.status_code == 200
    assert data["saved"] is True
    assert data["display_value"] == "••••••••••••"


def test_clean_markdown_artifacts_removes_dangling_bold_marker():
    text = _clean_markdown_artifacts("** 【扩展关联推理】\n分析范围")

    assert "**" not in text
    assert "【扩展关联推理】" in text


def test_deepseek_key_is_not_returned_in_response(monkeypatch):
    async def fake_generate(self, question, evidence, market_summary, news_summary, graph_summary):
        return {
            "answer": "基于证据，新闻与行情在时间上存在关联迹象。",
            "provider": "deepseek",
            "warnings": [],
        }

    monkeypatch.setattr("app.llm.DeepSeekLLMClient.generate", fake_generate)
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "最近一个月 TSLA 为什么波动？",
            "deepseek_api_key": "test-secret-key",
        },
    )

    assert response.status_code == 200
    assert "test-secret-key" not in response.text


def test_long_llm_report_body_is_preserved(monkeypatch):
    long_report = (
        "### 期刊式主体讨论\n"
        "第一段说明新闻主题、行业链条和行情窗口之间的关系。"
        "这里不是简单罗列新闻，而是解释为什么 AI、EV 和利率主题可能共同影响市场叙事。\n\n"
        "第二段进一步讨论企业之间的传导关系。上游算力供应商、下游应用企业和资本成本敏感行业"
        "可能共享同一组外部变量，因此新闻影响应被理解为跨行业的关联解释。\n\n"
        "第三段讨论局限性。现有证据只能说明时间重合、主题共振和关系路径假设，不能证明确定因果。"
    )

    class FakeLLM:
        async def generate(self, question, evidence, market_summary, news_summary, graph_summary):
            return {"answer": long_report, "provider": "deepseek", "warnings": []}

    monkeypatch.setattr("app.rag.get_llm_client", lambda settings, api_key_override=None: FakeLLM())
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "AI、EV 和利率新闻如何影响行业？",
            "deepseek_api_key": "test-secret-key",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert "期刊式主体讨论" in data["answer"]
    assert "第一段说明新闻主题" in data["answer"]
    assert "第二段进一步讨论企业之间的传导关系" in data["answer"]


def test_llm_failure_uses_specific_fallback_report(monkeypatch):
    class FailedLLM:
        async def generate(self, question, evidence, market_summary, news_summary, graph_summary):
            return {
                "answer": "已完成真实行情和新闻证据检索，但 DeepSeek 调用失败，无法生成 LLM 解释。",
                "provider": "deepseek_failed",
                "warnings": ["DeepSeek 调用失败"],
            }

    monkeypatch.setattr("app.rag.get_llm_client", lambda settings, api_key_override=None: FailedLLM())
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "AI、EV 和利率新闻如何影响行业？",
            "deepseek_api_key": "test-secret-key",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert "DeepSeek 调用失败，无法生成 LLM 解释" not in data["answer"]
    assert "新闻事实与主题介绍" in data["answer"]
    assert "情景推演" in data["answer"]


def test_save_deepseek_key_flag_calls_secret_writer(monkeypatch):
    saved = {}

    def fake_save(api_key):
        saved["api_key"] = api_key

    monkeypatch.setattr("app.rag.save_deepseek_api_key", fake_save)
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "最近一个月 TSLA 为什么波动？",
            "deepseek_api_key": "test-secret-key",
            "save_deepseek_api_key": True,
        },
    )

    assert response.status_code == 200
    assert saved["api_key"] == "test-secret-key"
    assert "test-secret-key" not in response.text


def test_analyze_with_no_evidence_does_not_fabricate(monkeypatch):
    class EmptyNewsSource:
        async def search_news(self, company_name, ticker, aliases, start_date, end_date, top_k):
            return []

    monkeypatch.setattr("app.rag.get_news_source", lambda settings: EmptyNewsSource())
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "TSLA",
            "company_name": "Tesla",
            "question": "为什么波动？",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["citations"] == []
    assert data["evaluation"]["citation_coverage"] == 0.0
    assert data["evaluation"]["coverage_status"]
    assert any(item["output_count"] == 0 for item in data["retrieval_trace"])


def test_mock_fixture_path_does_not_need_network():
    request = AnalyzeRequest(
        ticker="TSLA",
        company_name="Tesla",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        question="为什么波动？",
        top_k_news=100,
    )

    result = anyio.run(run_analysis, request, _test_settings(), None)

    assert result.evaluation["llm_provider"] == "mock"


def test_analysis_total_timeout_returns_partial_result(monkeypatch):
    class SlowMarketSource:
        async def get_prices(self, ticker, start_date, end_date):
            await anyio.sleep(2)
            return []

    settings = _test_settings()
    settings.analysis_total_timeout_seconds = 0.01
    monkeypatch.setattr("app.rag.get_market_source", lambda settings: SlowMarketSource())

    request = AnalyzeRequest(
        ticker="TSLA",
        company_name="Tesla",
        question="timeout test",
    )
    result = anyio.run(run_analysis, request, settings, None)

    assert result.evaluation["partial_result"] is True
    assert result.evaluation["llm_provider"] == "timeout_partial"
    assert result.evaluation["partial_reason"]


def test_real_deepseek_missing_key_does_not_return_mock_answer():
    settings = Settings()
    settings.use_mock_data = False
    settings.llm_provider = "deepseek"
    settings.deepseek_api_key = ""
    client_obj = DeepSeekLLMClient(settings)

    async def run():
        return await client_obj.generate(
            "为什么波动？",
            [{"title": "real article", "url": "https://example.com"}],
            {"market_data_days": 1, "abnormal_moves": []},
            {"news_count": 1, "event_window_news_count": 1},
            {},
        )

    result = anyio.run(run)

    assert result["provider"] == "deepseek_missing_key"
    assert "mock" not in result["answer"].lower()


def test_company_search_endpoint_returns_results(monkeypatch):
    async def fake_search_companies(q, settings, db):
        return [
            CompanySearchResult(
                ticker="TSLA",
                company_name="Tesla, Inc.",
                exchange="NASDAQ",
                aliases=["Tesla"],
                source="test",
            )
        ]

    monkeypatch.setattr("app.main.search_companies", fake_search_companies)
    response = client.get("/api/companies/search?q=tes")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["ticker"] == "TSLA"
