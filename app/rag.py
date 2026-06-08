import asyncio
import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any
import re

try:
    from sqlalchemy import and_, select
    from sqlalchemy.orm import Session
except ImportError:
    and_ = None
    select = None
    Session = Any

from .analytics import summarize_market
from .config import Settings
from .db import SQLALCHEMY_AVAILABLE
from .graph import build_graph_summary
from .guard import DISCLAIMER, apply_guard
from .insights import build_reasoning_addendum, build_reasoning_context
from .llm import get_llm_client
from .models import Analysis, Citation, MarketPrice, NewsArticle
from .schemas import AnalyzeRequest, AnalyzeResponse, CitationOut
from .secrets import save_deepseek_api_key
from .sources import SourceError, get_market_source, get_news_source


MIN_NEWS_CANDIDATES = 100
REASONING_CONTEXT_LIMIT = 50
LLM_EVIDENCE_LIMIT = 50
GRAPH_ARTICLE_LIMIT = 30
DISPLAY_CITATION_LIMIT = 50
ProgressCallback = Callable[[dict[str, Any]], None]


class AnalysisTimeout(RuntimeError):
    pass


async def run_analysis(
    request: AnalyzeRequest,
    settings: Settings,
    db: Session | None,
    progress: ProgressCallback | None = None,
) -> AnalyzeResponse:
    warnings: list[str] = []
    evaluation_notes: list[str] = []
    missing_real_data_reasons: list[str] = []
    total_timeout = float(settings.analysis_total_timeout_seconds)
    deadline = time.monotonic() + max(1.0, total_timeout) if total_timeout > 0 else None
    prices: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []

    def report(stage: str, percent: int, message: str, detail: dict[str, Any] | None = None) -> None:
        _report_progress(progress, stage, percent, message, detail)

    report("问题解析", 4, "正在解析问题、企业和时间窗口。")
    start_date, end_date, top_k_news, inference_reason = _infer_analysis_scope(request)
    selected_entities = _select_entities_from_question(request.question, request.ticker, request.company_name)
    primary_entity = selected_entities[0]
    entity_aliases = []
    for entity in selected_entities:
        entity_aliases.extend([entity["ticker"], entity["company_name"]])
    request = request.model_copy(
        update={
            "ticker": primary_entity["ticker"],
            "company_name": primary_entity["company_name"],
            "aliases": [*request.aliases, *entity_aliases],
        }
    )
    aliases = _expanded_news_aliases(request)
    report(
        "实体识别",
        10,
        "已识别主要企业和检索关键词。",
        {"ticker": request.ticker.upper(), "company_name": request.company_name, "aliases": aliases[:10]},
    )

    if request.deepseek_api_key and request.save_deepseek_api_key:
        save_deepseek_api_key(request.deepseek_api_key)
        settings.deepseek_api_key = request.deepseek_api_key

    retrieval_trace: list[dict[str, Any]] = [
        {
            "stage": "问题解析与参数推断",
            "input": {
                "question": request.question,
                "provided_start_date": str(request.start_date) if request.start_date else None,
                "provided_end_date": str(request.end_date) if request.end_date else None,
                "provided_top_k_news": request.top_k_news,
            },
            "output_count": top_k_news,
            "reason": inference_reason,
            "output": {"start_date": str(start_date), "end_date": str(end_date), "top_k_news": top_k_news},
        },
        {
            "stage": "实体与问题关键词识别",
            "input": {"ticker": request.ticker.upper(), "company_name": request.company_name, "aliases": request.aliases},
            "output_count": 2 + len(aliases),
            "reason": "使用股票代码、公司名、别名和问题关键词共同扩展新闻检索，支持企业关系、行业影响和具体新闻延伸分析。",
        },
    ]

    try:
        report("行情检索", 18, "正在读取缓存或请求真实行情源。", {"ticker": request.ticker.upper()})
        prices = await _with_budget(
            lambda: _get_prices(request, settings, db, start_date, end_date, warnings, evaluation_notes, missing_real_data_reasons),
            deadline,
            "行情检索",
        )
    except AnalysisTimeout as exc:
        return _build_partial_response(
            request,
            settings,
            db,
            start_date,
            end_date,
            top_k_news,
            prices,
            articles,
            retrieval_trace,
            warnings,
            evaluation_notes,
            missing_real_data_reasons,
            str(exc),
            progress,
        )
    report("行情检索", 28, "行情检索完成。", {"market_rows": len(prices)})
    retrieval_trace.append(
        {
            "stage": "行情检索",
            "input": {"ticker": request.ticker.upper(), "start_date": str(start_date), "end_date": str(end_date)},
            "output_count": len(prices),
            "reason": "优先复用 PostgreSQL 缓存；缓存不足时调用真实行情源。若真实源失败，则明确记录失败原因。",
        }
    )

    try:
        report("新闻检索", 34, "正在请求真实新闻源并去重。", {"target_candidates": top_k_news})
        articles = await _with_budget(
            lambda: _get_articles(request, settings, db, start_date, end_date, top_k_news, aliases, warnings, evaluation_notes, missing_real_data_reasons),
            deadline,
            "新闻检索",
        )
    except AnalysisTimeout as exc:
        return _build_partial_response(
            request,
            settings,
            db,
            start_date,
            end_date,
            top_k_news,
            prices,
            articles,
            retrieval_trace,
            warnings,
            evaluation_notes,
            missing_real_data_reasons,
            str(exc),
            progress,
        )
    report("新闻检索", 46, "新闻检索完成。", {"news_rows": len(articles)})
    retrieval_trace.append(
        {
            "stage": "新闻检索",
            "input": {
                "company_name": request.company_name,
                "ticker": request.ticker.upper(),
                "top_k_news": top_k_news,
                "provider": "mock" if settings.use_mock_data else settings.news_provider,
                "expanded_terms": aliases[:12],
            },
            "output_count": len(articles),
            "reason": "新闻候选数最低 100；同时保留与问题词、行业词、其他企业词相关但不一定直接命中目标公司的新闻。",
        }
    )

    market_summary = summarize_market(prices)
    retrieval_trace.append(
        {
            "stage": "异常波动日识别",
            "input": {"market_data_days": market_summary.get("market_data_days", 0)},
            "output_count": len(market_summary.get("abnormal_moves", [])),
            "reason": "按绝对日收益率排序，提取主要异动交易日。",
        }
    )

    ranked_candidates = _filter_and_rank_articles(
        articles,
        request.company_name,
        request.ticker,
        market_summary.get("abnormal_moves", []),
        top_k_news,
    )
    context_articles = ranked_candidates[:REASONING_CONTEXT_LIMIT]
    llm_articles = context_articles[:LLM_EVIDENCE_LIMIT]
    graph_articles = context_articles[:GRAPH_ARTICLE_LIMIT]
    citation_articles = context_articles[:DISPLAY_CITATION_LIMIT]
    report(
        "过滤重排",
        56,
        "已完成新闻重排和数量截断。",
        {
            "ranked_candidates": len(ranked_candidates),
            "reasoning_context_news": len(context_articles),
            "llm_evidence_news": len(llm_articles),
            "graph_news": len(graph_articles),
        },
    )
    retrieval_trace.append(
        {
            "stage": "时间窗口过滤、弱相关保留与重排序",
            "input": {
                "news_count": len(articles),
                "event_windows": [move["trade_date"] for move in market_summary.get("abnormal_moves", [])],
            },
            "output_count": len(ranked_candidates),
            "output": {
                "candidate_news": len(articles),
                "ranked_candidates": len(ranked_candidates),
                "kept_for_reasoning": len(context_articles),
                "llm_evidence_limit": len(llm_articles),
                "graph_article_limit": len(graph_articles),
                "citation_display_limit": len(citation_articles),
            },
            "reason": "优先保留异动日前后新闻，同时保留问题关键词、行业词和跨企业词命中的弱相关新闻，用于探索性关系推理。",
        }
    )

    reasoning_context = build_reasoning_context(
        request.question,
        context_articles or articles[:REASONING_CONTEXT_LIMIT],
        request.ticker,
        request.company_name,
    )
    try:
        report("企业行情补充", 62, "正在补充回答中涉及企业的行情表。")
        market_summary["company_market_summaries"] = await _with_budget(
            lambda: _get_company_market_summaries(
                selected_entities,
                reasoning_context.get("mentioned_companies", []),
                request,
                settings,
                db,
                start_date,
                end_date,
                prices,
                warnings,
                evaluation_notes,
                missing_real_data_reasons,
            ),
            deadline,
            "企业行情补充",
        )
    except AnalysisTimeout as exc:
        market_summary["company_market_summaries"] = []
        return _build_partial_response(
            request,
            settings,
            db,
            start_date,
            end_date,
            top_k_news,
            prices,
            articles,
            retrieval_trace,
            warnings,
            evaluation_notes,
            missing_real_data_reasons,
            str(exc),
            progress,
        )
    news_summary = {
        "news_count": len(articles),
        "ranked_candidate_news_count": len(ranked_candidates),
        "event_window_news_count": len(context_articles),
        "llm_evidence_news_count": len(llm_articles),
        "graph_news_count": len(graph_articles),
        "citation_news_count": len(citation_articles),
        "selected_titles": [article["title"] for article in context_articles[:8]],
        "reasoning_context": reasoning_context,
    }
    graph_summary = build_graph_summary(
        request.ticker.upper(),
        request.company_name,
        graph_articles,
        market_summary,
        reasoning_context,
    )
    citations = _articles_to_citations(citation_articles)
    report("图谱与引用", 72, "图谱、引用和推理上下文已生成。", {"citations": len(citations), "graph_news": len(graph_articles)})
    retrieval_trace.append(
        {
            "stage": "引用、行业映射与词云生成",
            "input": {
                "ranked_candidates": len(ranked_candidates),
                "reasoning_context_news": len(context_articles),
                "llm_evidence_news": len(llm_articles),
                "graph_news": len(graph_articles),
                "citation_news": len(citation_articles),
            },
            "output_count": len(citations),
            "reason": "生成可追溯引用，同时把新闻标题和摘要映射为词云、行业影响、企业关系和探索性因果路径。",
        }
    )

    evidence = [
        {
            "title": article["title"],
            "published_at": article["published_at"].isoformat(),
            "source_domain": article["source_domain"],
            "short_snippet": article.get("short_snippet") or "",
            "url": article["url"],
        }
        for article in llm_articles
    ]

    try:
        report("LLM 解释", 82, "正在调用 LLM 生成关联解释。", {"evidence_rows": len(evidence)})
        llm_result = await _with_budget(
            lambda: get_llm_client(settings, request.deepseek_api_key).generate(
                request.question,
                evidence,
                market_summary,
                news_summary,
                graph_summary,
            ),
            deadline,
            "LLM 解释",
        )
    except AnalysisTimeout as exc:
        return _build_partial_response(
            request,
            settings,
            db,
            start_date,
            end_date,
            top_k_news,
            prices,
            articles,
            retrieval_trace,
            warnings,
            evaluation_notes,
            missing_real_data_reasons,
            str(exc),
            progress,
        )
    warnings.extend(llm_result.get("warnings", []))
    answer = _format_answer(
        llm_result["answer"],
        request.question,
        market_summary,
        news_summary,
        reasoning_context,
        citations,
    )
    guarded = apply_guard(answer, warnings)

    evaluation = {
        "partial_result": False,
        "real_data_mode": not settings.use_mock_data,
        "citation_coverage": 0.0 if not citations else 1.0,
        "news_count": len(articles),
        "ranked_candidate_news_count": len(ranked_candidates),
        "event_window_news_count": len(context_articles),
        "llm_evidence_news_count": len(llm_articles),
        "graph_news_count": len(graph_articles),
        "citation_news_count": len(citation_articles),
        "market_data_days": market_summary.get("market_data_days", 0),
        "market_provider": prices[0]["data_source"] if prices else settings.market_provider,
        "news_provider": articles[0]["raw_source"] if articles else settings.news_provider,
        "llm_provider": llm_result.get("provider", settings.llm_provider),
        "embed_provider": settings.embed_provider,
        "analysis_scope": reasoning_context["analysis_scope"],
        "reasoning_mode": reasoning_context["reasoning_mode"],
        "causality_overclaim_risk": "medium",
        "investment_advice_risk": "low",
        "answer_confidence": _confidence(citations, prices, context_articles),
        "confidence_reasons": _confidence_reasons(citations, prices, context_articles),
        "coverage_status": _coverage_status(citations, prices, context_articles),
        "risk_status": "已通过风险守卫；扩展推理仅作为探索性关联假设，不作为确定因果或投资建议。",
        "data_coverage": {
            "requested_start_date": str(start_date),
            "requested_end_date": str(end_date),
            "market_days": market_summary.get("market_data_days", 0),
            "company_market_tables": len(market_summary.get("company_market_summaries", [])),
            "news_articles": len(articles),
            "ranked_candidate_news": len(ranked_candidates),
            "event_window_news": len(context_articles),
            "llm_evidence_news": len(llm_articles),
            "graph_news": len(graph_articles),
            "citation_news": len(citation_articles),
            "minimum_news_candidates": MIN_NEWS_CANDIDATES,
            "reasoning_context_limit": REASONING_CONTEXT_LIMIT,
            "llm_evidence_limit": LLM_EVIDENCE_LIMIT,
            "graph_article_limit": GRAPH_ARTICLE_LIMIT,
            "citation_display_limit": DISPLAY_CITATION_LIMIT,
        },
        "missing_real_data_reasons": missing_real_data_reasons,
        "notes": evaluation_notes,
    }

    _save_analysis(
        db,
        request,
        start_date,
        end_date,
        guarded["answer"],
        market_summary,
        news_summary,
        graph_summary,
        evaluation,
        guarded["risk_warnings"],
        citations,
    )

    report("完成", 100, "分析完成，正在返回完整结果。")
    return AnalyzeResponse(
        answer=guarded["answer"],
        claim_level="association_only",
        retrieval_trace=retrieval_trace,
        market_summary=market_summary,
        news_summary=news_summary,
        graph_summary=graph_summary,
        citations=citations,
        evaluation=evaluation,
        risk_warnings=guarded["risk_warnings"],
    )


def _report_progress(
    progress: ProgressCallback | None,
    stage: str,
    percent: int,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    if progress is None:
        return
    progress({"stage": stage, "percent": percent, "message": message, "detail": detail or {}})


async def _with_budget(factory: Callable[[], Any], deadline: float | None, stage: str) -> Any:
    if deadline is None:
        return await factory()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AnalysisTimeout(f"{stage} 超过总分析时间预算，已返回部分结果。")
    try:
        return await asyncio.wait_for(factory(), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise AnalysisTimeout(f"{stage} 超过总分析时间预算，已返回部分结果。") from exc


def _build_partial_response(
    request: AnalyzeRequest,
    settings: Settings,
    db: Session | None,
    start_date: date,
    end_date: date,
    top_k_news: int,
    prices: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    warnings: list[str],
    evaluation_notes: list[str],
    missing_real_data_reasons: list[str],
    partial_reason: str,
    progress: ProgressCallback | None,
) -> AnalyzeResponse:
    warnings = [*warnings, partial_reason]
    evaluation_notes = [*evaluation_notes, partial_reason]
    missing_real_data_reasons = [*missing_real_data_reasons, partial_reason]
    retrieval_trace = [
        *retrieval_trace,
        {
            "stage": "超时部分返回",
            "input": {"top_k_news": top_k_news},
            "output_count": len(articles),
            "reason": partial_reason,
        },
    ]

    market_summary = summarize_market(prices)
    market_summary["company_market_summaries"] = []
    ranked_candidates = _filter_and_rank_articles(
        articles,
        request.company_name,
        request.ticker,
        market_summary.get("abnormal_moves", []),
        top_k_news,
    )
    context_articles = ranked_candidates[:REASONING_CONTEXT_LIMIT]
    llm_articles = context_articles[:LLM_EVIDENCE_LIMIT]
    graph_articles = context_articles[:GRAPH_ARTICLE_LIMIT]
    citation_articles = context_articles[:DISPLAY_CITATION_LIMIT]
    reasoning_context = build_reasoning_context(
        request.question,
        context_articles or articles[:REASONING_CONTEXT_LIMIT],
        request.ticker,
        request.company_name,
    )
    news_summary = {
        "news_count": len(articles),
        "ranked_candidate_news_count": len(ranked_candidates),
        "event_window_news_count": len(context_articles),
        "llm_evidence_news_count": len(llm_articles),
        "graph_news_count": len(graph_articles),
        "citation_news_count": len(citation_articles),
        "selected_titles": [article["title"] for article in context_articles[:8]],
        "reasoning_context": reasoning_context,
    }
    graph_summary = build_graph_summary(
        request.ticker.upper(),
        request.company_name,
        graph_articles,
        market_summary,
        reasoning_context,
    )
    citations = _articles_to_citations(citation_articles)
    llm_result = {
        "answer": f"分析达到时间上限，系统已基于当前已取得的证据返回部分结果。\n原因：{partial_reason}\n\n{DISCLAIMER}",
        "provider": "timeout_partial",
        "warnings": [partial_reason],
    }
    answer = _format_answer(
        llm_result["answer"],
        request.question,
        market_summary,
        news_summary,
        reasoning_context,
        citations,
    )
    guarded = apply_guard(answer, warnings)
    evaluation = {
        "partial_result": True,
        "partial_reason": partial_reason,
        "real_data_mode": not settings.use_mock_data,
        "citation_coverage": 0.0 if not citations else 1.0,
        "news_count": len(articles),
        "ranked_candidate_news_count": len(ranked_candidates),
        "event_window_news_count": len(context_articles),
        "llm_evidence_news_count": len(llm_articles),
        "graph_news_count": len(graph_articles),
        "citation_news_count": len(citation_articles),
        "market_data_days": market_summary.get("market_data_days", 0),
        "market_provider": prices[0]["data_source"] if prices else settings.market_provider,
        "news_provider": articles[0]["raw_source"] if articles else settings.news_provider,
        "llm_provider": llm_result["provider"],
        "embed_provider": settings.embed_provider,
        "analysis_scope": reasoning_context["analysis_scope"],
        "reasoning_mode": reasoning_context["reasoning_mode"],
        "causality_overclaim_risk": "medium",
        "investment_advice_risk": "low",
        "answer_confidence": _confidence(citations, prices, context_articles),
        "confidence_reasons": _confidence_reasons(citations, prices, context_articles),
        "coverage_status": _coverage_status(citations, prices, context_articles),
        "risk_status": "已返回部分结果；输出仍限定为 association_only，不构成投资建议。",
        "data_coverage": {
            "requested_start_date": str(start_date),
            "requested_end_date": str(end_date),
            "market_days": market_summary.get("market_data_days", 0),
            "company_market_tables": 0,
            "news_articles": len(articles),
            "ranked_candidate_news": len(ranked_candidates),
            "event_window_news": len(context_articles),
            "llm_evidence_news": len(llm_articles),
            "graph_news": len(graph_articles),
            "citation_news": len(citation_articles),
            "minimum_news_candidates": MIN_NEWS_CANDIDATES,
            "reasoning_context_limit": REASONING_CONTEXT_LIMIT,
            "llm_evidence_limit": LLM_EVIDENCE_LIMIT,
            "graph_article_limit": GRAPH_ARTICLE_LIMIT,
            "citation_display_limit": DISPLAY_CITATION_LIMIT,
        },
        "missing_real_data_reasons": missing_real_data_reasons,
        "notes": evaluation_notes,
    }
    _save_analysis(
        db,
        request,
        start_date,
        end_date,
        guarded["answer"],
        market_summary,
        news_summary,
        graph_summary,
        evaluation,
        guarded["risk_warnings"],
        citations,
    )
    _report_progress(progress, "部分结果", 100, "达到时间上限，已返回部分结果。", {"partial_reason": partial_reason})
    return AnalyzeResponse(
        answer=guarded["answer"],
        claim_level="association_only",
        retrieval_trace=retrieval_trace,
        market_summary=market_summary,
        news_summary=news_summary,
        graph_summary=graph_summary,
        citations=citations,
        evaluation=evaluation,
        risk_warnings=guarded["risk_warnings"],
    )


async def _get_company_market_summaries(
    selected_entities: list[dict[str, str]],
    mentioned_companies: list[dict[str, str]],
    request: AnalyzeRequest,
    settings: Settings,
    db: Session | None,
    start_date: date,
    end_date: date,
    primary_prices: list[dict[str, Any]],
    warnings: list[str],
    evaluation_notes: list[str],
    missing_real_data_reasons: list[str],
) -> list[dict[str, Any]]:
    entities = _dedupe_entities([*selected_entities, *mentioned_companies])
    summaries: list[dict[str, Any]] = []
    for entity in entities[: max(1, settings.related_market_entity_limit)]:
        ticker = entity["ticker"].upper()
        if ticker == request.ticker.upper():
            prices = primary_prices
        else:
            entity_request = request.model_copy(
                update={"ticker": ticker, "company_name": entity.get("company_name") or ticker}
            )
            prices = await _get_prices(
                entity_request,
                settings,
                db,
                start_date,
                end_date,
                warnings,
                evaluation_notes,
                missing_real_data_reasons,
            )
        summaries.append(_summarize_company_market(entity, prices))
    return summaries


def _dedupe_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    results = []
    for entity in entities:
        ticker = (entity.get("ticker") or "").upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        results.append({"ticker": ticker, "company_name": entity.get("company_name") or ticker})
    return results


def _summarize_company_market(entity: dict[str, str], prices: list[dict[str, Any]]) -> dict[str, Any]:
    base = {
        "ticker": entity["ticker"].upper(),
        "company_name": entity.get("company_name") or entity["ticker"].upper(),
        "market_data_days": 0,
        "start_date": None,
        "end_date": None,
        "latest_close": None,
        "total_return": None,
        "max_abs_daily_return": None,
        "abnormal_move_count": 0,
        "data_source": None,
        "status": "missing",
        "annotation": "未取得该企业在当前窗口内的行情数据。",
    }
    if not prices:
        return base

    ordered = sorted(prices, key=lambda item: str(item["trade_date"]))
    first_close = float(ordered[0]["close"])
    last_close = float(ordered[-1]["close"])
    summary = summarize_market(ordered)
    abnormal_moves = summary.get("abnormal_moves", [])
    total_return = None if first_close == 0 else round(last_close / first_close - 1, 6)
    max_abs = max([abs(item["daily_return"]) for item in summary.get("daily_returns", [])] or [0.0])
    base.update(
        {
            "market_data_days": len(ordered),
            "start_date": str(ordered[0]["trade_date"]),
            "end_date": str(ordered[-1]["trade_date"]),
            "latest_close": round(last_close, 4),
            "total_return": total_return,
            "max_abs_daily_return": round(max_abs, 6),
            "abnormal_move_count": len(abnormal_moves),
            "data_source": ordered[0].get("data_source"),
            "status": "ok",
            "annotation": (
                f"{entity.get('company_name') or entity['ticker']} 在当前窗口覆盖 {len(ordered)} 个交易日，"
                f"区间收益率为 {_pct(total_return)}，最大单日绝对波动为 {_pct(max_abs)}。"
            ),
        }
    )
    return base


def _pct(value: float | None) -> str:
    if value is None:
        return "暂无"
    return f"{value * 100:.2f}%"


async def _get_prices(
    request: AnalyzeRequest,
    settings: Settings,
    db: Session | None,
    start_date: date,
    end_date: date,
    warnings: list[str],
    evaluation_notes: list[str],
    missing_real_data_reasons: list[str],
) -> list[dict[str, Any]]:
    cached_prices = _load_cached_prices(db, request.ticker, start_date, end_date, allow_mock=settings.use_mock_data)
    prices = cached_prices if _cache_covers_range(cached_prices, start_date, end_date) else []
    if prices:
        return prices
    try:
        prices = await get_market_source(settings).get_prices(request.ticker, start_date, end_date)
        _save_prices(db, prices)
        return prices
    except SourceError as exc:
        if settings.use_mock_data:
            from .sources import MockMarketSource

            warnings.append(f"外部行情源失败，测试模式降级到 mock 行情：{exc}")
            return await MockMarketSource().get_prices(request.ticker, start_date, end_date)
        warnings.append(f"真实行情源失败：{exc}")
        missing_real_data_reasons.append(f"真实行情源失败：{exc}")
        evaluation_notes.append(str(exc))
        return []


async def _get_articles(
    request: AnalyzeRequest,
    settings: Settings,
    db: Session | None,
    start_date: date,
    end_date: date,
    top_k_news: int,
    aliases: list[str],
    warnings: list[str],
    evaluation_notes: list[str],
    missing_real_data_reasons: list[str],
) -> list[dict[str, Any]]:
    try:
        articles = await get_news_source(settings).search_news(
            request.company_name,
            request.ticker,
            aliases,
            start_date,
            end_date,
            top_k_news,
        )
        _save_articles(db, articles)
        return articles
    except SourceError as exc:
        if settings.use_mock_data:
            from .sources import MockNewsSource

            warnings.append(f"外部新闻源失败，测试模式降级到 mock 新闻：{exc}")
            return await MockNewsSource().search_news(
                request.company_name,
                request.ticker,
                aliases,
                start_date,
                end_date,
                top_k_news,
            )
        warnings.append(f"真实新闻源失败：{exc}")
        missing_real_data_reasons.append(f"真实新闻源失败：{exc}")
        evaluation_notes.append(str(exc))
        return []


def _infer_analysis_scope(request: AnalyzeRequest) -> tuple[date, date, int, str]:
    today = date.today()
    question = request.question.lower()
    end_date = request.end_date or today
    if request.start_date:
        start_date = request.start_date
        window_days = max((end_date - start_date).days, 1)
        date_reason = "使用请求中显式提供的时间范围。"
    else:
        window_days = _infer_window_days(question)
        start_date = end_date - timedelta(days=window_days)
        date_reason = f"根据问题语义自动选择最近 {window_days} 天作为新闻和行情检索窗口。"
    if start_date > end_date:
        start_date, end_date = end_date, start_date
        date_reason += " 检测到开始日期晚于结束日期，已自动交换。"
    inferred_top_k = _infer_news_limit(window_days)
    top_k_news = max(request.top_k_news if request.top_k_news is not None else inferred_top_k, MIN_NEWS_CANDIDATES)
    return start_date, end_date, top_k_news, f"{date_reason} 新闻候选数量至少为 100，当前选择 {top_k_news} 条。"


def _infer_window_days(question: str) -> int:
    if any(token in question for token in ["今天", "今日", "当天"]):
        return 7
    if any(token in question for token in ["一周", "1周", "7天", "七天", "最近几天", "这几天"]):
        return 14
    if any(token in question for token in ["三个月", "3个月", "90天", "季度"]):
        return 100
    if any(token in question for token in ["半年", "6个月", "六个月"]):
        return 190
    return 45


def _infer_news_limit(window_days: int) -> int:
    if window_days >= 180:
        return 160
    if window_days >= 90:
        return 130
    return 100


def _expanded_news_aliases(request: AnalyzeRequest) -> list[str]:
    words = []
    for token in request.question.replace("，", " ").replace("。", " ").replace("？", " ").split():
        clean = token.strip(" ,.?;:!()[]{}\"'")
        if (
            2 <= len(clean) <= 32
            and clean.lower() not in {"why", "what", "how", "recent", "latest"}
            and not re.search(r"[\u4e00-\u9fff]", clean)
        ):
            words.append(clean)
    return [*request.aliases, *words[:12]]


def _strip_llm_preface(answer: str) -> str:
    blocked_prefixes = [
        "好的，作为金融信息分析助手，",
        "好的，",
        "作为金融信息分析助手，",
    ]
    cleaned = answer.strip()
    for prefix in blocked_prefixes:
        if cleaned.startswith(prefix):
            first_break = cleaned.find("\n")
            first_period = cleaned.find("。")
            cut_points = [pos for pos in [first_break, first_period] if pos != -1]
            if cut_points:
                cleaned = cleaned[min(cut_points) + 1 :].strip()
            else:
                cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _clean_markdown_artifacts(answer: str) -> str:
    lines = []
    for line in answer.splitlines():
        # Remove dangling bold markers such as "** 【扩展关联推理】".
        if line.count("**") % 2 == 1:
            line = line.replace("**", "")
        lines.append(line)
    return "\n".join(lines).replace("** 【", "【").replace("**【", "【")


def _format_answer(
    llm_answer: str,
    question: str,
    market_summary: dict[str, Any],
    news_summary: dict[str, Any],
    reasoning_context: dict[str, Any],
    citations: list[CitationOut],
) -> str:
    cleaned = _clean_markdown_artifacts(_strip_llm_preface(llm_answer).replace(DISCLAIMER, "").strip())
    industries = reasoning_context.get("industry_impacts", [])[:4]
    hypotheses = reasoning_context.get("causal_hypotheses", [])[:3]
    relationships = reasoning_context.get("company_relationships", [])[:4]
    companies = reasoning_context.get("mentioned_companies", [])[:6]
    scope = reasoning_context.get("analysis_scope", "single_company")
    refs = _citation_markers(citations)
    body = _normalize_report_body(cleaned)
    if _is_llm_failure_body(body):
        body = ""
    if _is_thin_answer(body):
        fallback_body = _fallback_report_body(
            scope,
            industries,
            relationships,
            hypotheses,
            companies,
            citations,
            market_summary,
            refs,
        )
        body = f"{body}\n\n{fallback_body}" if body else fallback_body

    lines = [
        "## 题目",
        f"{question}：新闻舆情、企业关系与行情窗口的关联解释报告",
        "",
        "## 摘要",
        (
            f"本文围绕“{question}”构建新闻舆情-行情异动关联解释。系统识别到分析范围为 "
            f"**{scope}**，共检索到 **{news_summary.get('news_count', 0)}** 条新闻候选，"
            f"其中 **{news_summary.get('event_window_news_count', 0)}** 条进入推理上下文，"
            f"行情覆盖 **{market_summary.get('market_data_days', 0)}** 个交易日。"
            "综合新闻主题、企业关系、行业映射和行情窗口，当前材料支持的是"
            f"**关联解释与探索性因果路径假设**{refs}，但不能证明确定因果。"
        ),
        "",
        "## 材料与方法",
        (
            "本次分析采用多阶段检索流程：先从问题中识别企业、行业和主题词，再抓取近期真实新闻与行情数据；"
            "随后按新闻时间、主题相关性、企业命中和行情异动窗口进行重排；最后将进入推理上下文的材料交给 LLM "
            "生成解释，并由风险守卫过滤投资建议、目标价和确定性因果表述。"
        ),
        "",
        "## 主要发现",
        _core_explanation(scope, industries, companies, refs),
        "",
        "## 主体分析",
        body,
    ]

    if relationships:
        lines.extend(["", "## 企业关系与推理路径"])
        lines.extend(f"- {item}" for item in relationships)

    if industries:
        lines.extend(["", "## 行业影响摘要"])
        for item in industries:
            example_companies = "、".join(item.get("example_companies", [])[:4])
            keywords = "、".join(item.get("matched_keywords", [])[:5])
            lines.append(
                f"- **{item['industry']}**：关键词 {keywords or '暂无'}；"
                f"样例企业：{example_companies or '暂无'}。"
            )

    if hypotheses:
        lines.extend(["", "## 探索性因果路径"])
        lines.extend(f"- {item}" for item in hypotheses)

    lines.extend(
        [
            "",
            "## 证据强度与局限",
            (
                "以上结论主要依赖新闻标题、短摘要、来源域名、发布时间、行情窗口和结构化关键词之间的对应关系。"
                "它能够说明“哪些新闻主题可能与哪些行业或企业关系同时出现”，但不能排除宏观利率、市场风格、"
                "财报预期、资金流动和其他未纳入数据源的共同变量。"
            ),
        ]
    )

    lines.extend(["", "## 新闻证据"])
    if citations:
        for idx, citation in enumerate(citations[:DISPLAY_CITATION_LIMIT], start=1):
            lines.append(
                f"- **证据 {idx}**（{citation.source_domain}，{citation.published_at.date()}）："
                f"{citation.citation_text}"
            )
    else:
        lines.append("- 当前没有可引用新闻证据。")

    lines.extend(["", "## 风险边界", DISCLAIMER])
    return _clean_markdown_artifacts("\n".join(lines))


def _normalize_report_body(answer: str) -> str:
    if not answer:
        return ""
    return answer.strip()


def _is_thin_answer(answer: str) -> bool:
    text = re.sub(r"\s+", "", answer or "")
    return len(text) < 450 or answer.count("\n") < 4


def _is_llm_failure_body(answer: str) -> bool:
    failure_markers = [
        "DeepSeek 调用失败",
        "DeepSeek API key 未配置",
        "无法生成 LLM 解释",
        "当前证据不足",
        "LLM 调用失败",
    ]
    return any(marker in (answer or "") for marker in failure_markers)


def _fallback_report_body(
    scope: str,
    industries: list[dict[str, Any]],
    relationships: list[str],
    hypotheses: list[str],
    companies: list[dict[str, Any]],
    citations: list[CitationOut],
    market_summary: dict[str, Any],
    refs: str,
) -> str:
    industry_names = "、".join(item.get("industry", "") for item in industries[:3]) or "相关行业"
    company_names = "、".join(item.get("company_name", item.get("ticker", "")) for item in companies[:5]) or "相关企业"
    evidence_groups = _group_citations_by_theme(citations)
    market_sentence = _market_snapshot_sentence(market_summary)
    lines = [
        "### 研究式解释",
        (
            f"从当前材料看，本问题更接近 **{scope}** 类型分析。新闻文本中的高频主题并不是孤立信号，"
            f"而是通过行业链条、企业竞争关系、供应链联系和投资者风险偏好共同作用于市场解释框架。"
            f"因此，较稳妥的理解方式不是把某一条新闻视为行情变化的单一原因，而是观察新闻主题是否在"
            f"{industry_names} 等领域形成重复出现的叙事，并进一步观察这些叙事是否与 {company_names} "
            f"等企业的行情窗口发生时间重合{refs}。"
        ),
        market_sentence,
        "",
        "### 新闻事实与主题介绍",
    ]
    if evidence_groups:
        for group in evidence_groups:
            lines.append(f"**{group['label']}**：{group['summary']}")
            for item in group["items"][:3]:
                lines.append(f"- 证据[{item['index']}] {item['date']}，{item['source']}：{item['title']}")
    else:
        lines.append("当前没有可引用新闻标题，系统只能保留行情窗口和行业词汇层面的低置信度解释。")

    lines.extend(
        [
            "",
            "### 解释机制",
        ]
    )
    lines.extend(
        [
        (
            "第一，新闻关键词可以被视为市场关注点的代理变量。例如 AI、芯片、EV、利率、能源、材料等词汇，"
            "分别对应算力供给、终端需求、融资成本、生产要素价格和产业链约束。若这些主题在短期内集中出现，"
            "它们可能改变投资者对行业景气度、利润弹性和风险暴露的理解。"
        ),
        (
            "第二，企业之间的关系提供了新闻外溢的路径。上游供应商、下游应用企业、同业竞争者和平台生态伙伴"
            "可能共享同一组宏观变量或产业变量，因此一家公司相关的新闻并不只影响该公司叙事，也可能被市场"
            "迁移到同链条或同主题企业上。"
        ),
        (
            "第三，行情异动窗口提供了时间参照。若新闻集中出现的日期与市场波动窗口相近，系统只能认为两者"
            "存在可讨论的时间重合和主题共振，而不能据此断言新闻造成了价格变化。"
        ),
        ]
    )
    scenario_lines = _scenario_lines(evidence_groups, industries)
    if scenario_lines:
        lines.extend(["", "### 情景推演（非股价预测）"])
        lines.extend(f"- {item}" for item in scenario_lines)
        lines.append(
            "这些推演只描述新闻叙事可能影响行业关注度和风险偏好的方向，不给出价格预测、目标价或交易信号。"
        )
    if relationships:
        lines.extend(["", "### 企业关系解释"])
        lines.extend(f"- {item}" for item in relationships[:4])
    if industries:
        lines.extend(["", "### 行业分层解释"])
        for item in industries[:4]:
            lines.append(f"- **{item['industry']}**：{item.get('hypothesized_impact', '')}")
    if hypotheses:
        lines.extend(["", "### 可检验假设"])
        lines.extend(f"- {item}" for item in hypotheses[:4])
    return "\n\n".join(lines)


def _group_citations_by_theme(citations: list[CitationOut]) -> list[dict[str, Any]]:
    themes = [
        {
            "label": "AI 芯片、半导体与科技平台",
            "keywords": ["ai", "chip", "semiconductor", "tech", "software", "cloud", "data center", "nvidia", "amd"],
            "summary": "这些新闻用于观察算力需求、科技平台估值叙事和半导体景气度是否形成共同主题。",
        },
        {
            "label": "EV、能源、材料与供应链",
            "keywords": ["ev", "electric", "battery", "lithium", "energy", "oil", "cleantech", "supply", "mining"],
            "summary": "这些新闻用于观察电动车、储能、能源价格和关键材料供给之间的成本与需求联动。",
        },
        {
            "label": "利率、ETF、资金风格与大盘风险偏好",
            "keywords": ["rate", "fed", "yield", "etf", "s&p", "dow", "futures", "dividend", "value", "market"],
            "summary": "这些新闻用于观察利率敏感资产、指数 ETF、价值风格和市场风险偏好变化。",
        },
        {
            "label": "企业事件、监管与个股外溢",
            "keywords": ["earnings", "revenue", "regulatory", "collaboration", "payment", "offer", "shares", "stock"],
            "summary": "这些新闻用于观察企业经营事件、监管事项和合作消息是否外溢到同行或上下游企业。",
        },
    ]
    grouped: list[dict[str, Any]] = []
    used_indexes: set[int] = set()
    for theme in themes:
        items = []
        for idx, citation in enumerate(citations, start=1):
            text = f"{citation.citation_text} {citation.source_domain}".lower()
            if idx in used_indexes:
                continue
            if any(keyword in text for keyword in theme["keywords"]):
                items.append(
                    {
                        "index": idx,
                        "date": citation.published_at.date().isoformat(),
                        "source": citation.source_domain,
                        "title": citation.citation_text,
                    }
                )
        if items:
            used_indexes.update(item["index"] for item in items[:4])
            grouped.append({**theme, "items": items[:4]})
    if citations and not grouped:
        grouped.append(
            {
                "label": "未明显归类的新闻候选",
                "summary": "这些新闻暂未命中预设行业关键词，只能作为低置信度背景材料。",
                "items": [
                    {
                        "index": idx,
                        "date": citation.published_at.date().isoformat(),
                        "source": citation.source_domain,
                        "title": citation.citation_text,
                    }
                    for idx, citation in enumerate(citations[:6], start=1)
                ],
            }
        )
    return grouped


def _market_snapshot_sentence(market_summary: dict[str, Any]) -> str:
    moves = market_summary.get("abnormal_moves", []) or []
    if not market_summary.get("market_data_days"):
        return "行情侧当前没有可用交易日数据，因此走势判断只能依赖新闻侧主题分布，置信度较低。"
    if not moves:
        return (
            f"行情侧覆盖 {market_summary.get('market_data_days')} 个交易日，但未识别到显著异动日；"
            "这意味着新闻主题更多用于解释市场叙事，而不是对应明确的波动窗口。"
        )
    move_text = "；".join(
        f"{move.get('trade_date')} 日收益率 {float(move.get('daily_return', 0)) * 100:.2f}%"
        for move in moves[:3]
    )
    return (
        f"行情侧覆盖 {market_summary.get('market_data_days')} 个交易日，主要异动窗口包括 {move_text}。"
        "这些日期可作为新闻事件线与市场走势共同观察的时间锚点。"
    )


def _scenario_lines(evidence_groups: list[dict[str, Any]], industries: list[dict[str, Any]]) -> list[str]:
    labels = {group["label"] for group in evidence_groups}
    lines = []
    if "AI 芯片、半导体与科技平台" in labels:
        lines.append(
            "如果 AI 芯片、半导体 ETF 或科技平台新闻继续密集出现，后续市场叙事可能更偏向算力资本开支、数据中心需求和科技成长风格；相关企业样例包括 NVIDIA、AMD、TSMC、Microsoft、Alphabet。"
        )
    if "EV、能源、材料与供应链" in labels:
        lines.append(
            "如果锂、能源、电动车和供应链新闻继续升温，后续解释重点可能转向电池材料成本、充电基础设施、整车需求和能源约束；相关企业样例包括 Tesla、BYD、Rivian、NIO、Albemarle、Exxon Mobil。"
        )
    if "利率、ETF、资金风格与大盘风险偏好" in labels:
        lines.append(
            "如果 ETF、利率、价值股和指数期货类新闻占比上升，行业影响可能更多经由资金风格切换和风险偏好变化扩散，而不是只作用于单家公司。"
        )
    if "企业事件、监管与个股外溢" in labels:
        lines.append(
            "如果企业监管、业绩或合作事件继续出现，系统会优先观察这些事件是否沿竞争关系、上下游供应链或平台生态传导到相关企业。"
        )
    if not lines and industries:
        for item in industries[:3]:
            lines.append(f"{item['industry']} 后续可观察关键词包括 {', '.join(item.get('matched_keywords', [])[:4])}，用于判断新闻主题是否持续聚集。")
    return lines[:5]


def _citation_markers(citations: list[CitationOut], limit: int = 4) -> str:
    return "".join(f"^[{idx}]" for idx in range(1, min(len(citations), limit) + 1))


def _core_explanation(
    scope: str,
    industries: list[dict[str, Any]],
    companies: list[dict[str, Any]],
    refs: str,
) -> str:
    company_names = "、".join(item.get("company_name", item.get("ticker", "")) for item in companies[:5])
    industry_names = "、".join(item.get("industry", "") for item in industries[:3])
    ref = refs[:8]
    if scope in {"industry_period_impact", "news_industry_impact"}:
        target = industry_names or "多个行业"
        return (
            f"本次问题更接近跨行业影响分析。新闻中的高频主题会先映射到 {target}，"
            f"再通过企业样例、供应链、需求侧变化和风险偏好变化形成关联解释{ref}。"
        )
    if scope in {"company_relationship", "multi_company_news"}:
        target = company_names or "多家企业"
        return (
            f"本次问题更接近企业关系分析。系统将 {target} 放在同一新闻和行情窗口中比较，"
            f"重点观察竞争、上下游、平台生态和投资者情绪的共同变化{ref}。"
        )
    target = company_names or "目标企业"
    return (
        f"本次问题更接近单企业新闻-行情关联分析。系统围绕 {target} 的近期新闻、"
        f"行情异动日和行业关键词，给出时间重合与主题共振层面的解释{ref}。"
    )


KNOWN_COMPANIES = [
    {"ticker": "TSLA", "company_name": "Tesla", "aliases": ["tesla", "特斯拉", "tsla"]},
    {"ticker": "NVDA", "company_name": "NVIDIA", "aliases": ["nvidia", "英伟达", "nvda"]},
    {"ticker": "AAPL", "company_name": "Apple", "aliases": ["apple", "苹果", "aapl"]},
    {"ticker": "MSFT", "company_name": "Microsoft", "aliases": ["microsoft", "微软", "msft"]},
    {"ticker": "GOOGL", "company_name": "Alphabet", "aliases": ["alphabet", "google", "谷歌", "googl"]},
    {"ticker": "AMZN", "company_name": "Amazon", "aliases": ["amazon", "亚马逊", "amzn"]},
    {"ticker": "META", "company_name": "Meta", "aliases": ["meta", "facebook", "脸书"]},
    {"ticker": "AMD", "company_name": "AMD", "aliases": ["amd"]},
    {"ticker": "INTC", "company_name": "Intel", "aliases": ["intel", "英特尔", "intc"]},
    {"ticker": "RIVN", "company_name": "Rivian", "aliases": ["rivian", "rivn"]},
    {"ticker": "NIO", "company_name": "NIO", "aliases": ["nio", "蔚来"]},
    {"ticker": "JPM", "company_name": "JPMorgan", "aliases": ["jpmorgan", "摩根大通", "jpm"]},
]


def _select_entities_from_question(question: str, fallback_ticker: str, fallback_company: str) -> list[dict[str, str]]:
    text = question.lower()
    selected = []
    for item in KNOWN_COMPANIES:
        if any(alias.lower() in text for alias in item["aliases"]):
            selected.append({"ticker": item["ticker"], "company_name": item["company_name"]})
    if not selected and fallback_ticker and fallback_ticker.upper() != "SPY":
        selected.append({"ticker": fallback_ticker.upper(), "company_name": fallback_company or fallback_ticker.upper()})
    if not selected:
        selected.append({"ticker": "SPY", "company_name": "Broad Market"})
    seen = set()
    deduped = []
    for item in selected:
        if item["ticker"] in seen:
            continue
        seen.add(item["ticker"])
        deduped.append(item)
    return deduped


def _load_cached_prices(
    db: Session | None,
    ticker: str,
    start_date: date,
    end_date: date,
    allow_mock: bool = False,
) -> list[dict[str, Any]]:
    if db is None or not SQLALCHEMY_AVAILABLE or select is None:
        return []
    try:
        conditions = [
            MarketPrice.ticker == ticker.upper(),
            MarketPrice.trade_date >= start_date,
            MarketPrice.trade_date <= end_date,
        ]
        if not allow_mock:
            conditions.append(MarketPrice.data_source != "mock")
        rows = db.execute(select(MarketPrice).where(and_(*conditions))).scalars().all()
        return [
            {
                "ticker": row.ticker,
                "trade_date": row.trade_date.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
                "adjusted_close": float(row.adjusted_close) if row.adjusted_close is not None else None,
                "data_source": row.data_source,
            }
            for row in rows
        ]
    except Exception:
        db.rollback()
        return []


def _cache_covers_range(prices: list[dict[str, Any]], start_date: date, end_date: date) -> bool:
    if not prices:
        return False
    dates = [date.fromisoformat(str(price["trade_date"])) for price in prices]
    return min(dates) <= start_date and max(dates) >= end_date


def _save_prices(db: Session | None, prices: list[dict[str, Any]]) -> None:
    if db is None or not prices or not SQLALCHEMY_AVAILABLE or select is None:
        return
    try:
        for price in prices:
            trade_date = date.fromisoformat(str(price["trade_date"]))
            exists = db.execute(
                select(MarketPrice.id).where(
                    and_(
                        MarketPrice.ticker == price["ticker"].upper(),
                        MarketPrice.trade_date == trade_date,
                        MarketPrice.data_source == price["data_source"],
                    )
                )
            ).first()
            if exists:
                continue
            db.add(
                MarketPrice(
                    ticker=price["ticker"].upper(),
                    trade_date=trade_date,
                    open=price["open"],
                    high=price["high"],
                    low=price["low"],
                    close=price["close"],
                    volume=price["volume"],
                    adjusted_close=price.get("adjusted_close"),
                    data_source=price["data_source"],
                )
            )
        db.commit()
    except Exception:
        db.rollback()


def _save_articles(db: Session | None, articles: list[dict[str, Any]]) -> None:
    if db is None or not articles or not SQLALCHEMY_AVAILABLE or select is None:
        return
    try:
        for article in articles:
            exists = db.execute(
                select(NewsArticle.id).where(
                    and_(
                        NewsArticle.ticker == article["ticker"].upper(),
                        NewsArticle.url == article["url"],
                    )
                )
            ).first()
            if exists:
                continue
            db.add(
                NewsArticle(
                    ticker=article["ticker"].upper(),
                    title=article["title"],
                    url=article["url"],
                    source_domain=article["source_domain"],
                    published_at=article["published_at"],
                    tone=article.get("tone"),
                    language=article.get("language"),
                    short_snippet=(article.get("short_snippet") or "")[:300],
                    raw_source=article["raw_source"],
                )
            )
        db.commit()
    except Exception:
        db.rollback()


def _filter_and_rank_articles(articles, company_name, ticker, abnormal_moves, top_k):
    if not articles:
        return []
    event_dates = [date.fromisoformat(move["trade_date"]) for move in abnormal_moves]
    title_counts = {}
    ranked = []
    for article in articles:
        published_date = article["published_at"].date()
        text = f"{article.get('title', '')} {article.get('short_snippet', '')}".lower()
        distances = [abs((published_date - event_date).days) for event_date in event_dates] or [999]
        min_distance = min(distances)
        in_window = min_distance <= 3
        title_key = article["title"].strip().lower()
        title_counts[title_key] = title_counts.get(title_key, 0) + 1
        contains_entity = company_name.lower() in text or ticker.lower() in text
        broad_signal = any(token in text for token in ["ai", "chip", "ev", "rate", "supply", "demand", "cloud", "energy"])
        score = 0
        score += 50 if in_window else 0
        score += max(0, 20 - min_distance)
        score += 20 if contains_entity else 0
        score += 8 if broad_signal else 0
        score += 5 if article.get("short_snippet") else 0
        score -= 10 * (title_counts[title_key] - 1)
        ranked.append((score, article))
    selected = [item[1] for item in sorted(ranked, key=lambda item: item[0], reverse=True)]
    return selected[:top_k]


def _articles_to_citations(articles: list[dict[str, Any]]) -> list[CitationOut]:
    return [
        CitationOut(
            citation_text=(article.get("short_snippet") or article["title"])[:300],
            url=article["url"],
            source_domain=article["source_domain"],
            published_at=article["published_at"],
        )
        for article in articles
    ]


def _confidence(citations, prices, ranked_articles) -> str:
    if not citations or not prices:
        return "low"
    if len(ranked_articles) >= 10 and len(prices) >= 10:
        return "high"
    return "medium"


def _confidence_reasons(citations, prices, ranked_articles) -> list[str]:
    reasons = [
        f"引用数量：{len(citations)}",
        f"行情交易日数量：{len(prices)}",
        f"进入推理上下文的新闻数量：{len(ranked_articles)}",
    ]
    if not citations:
        reasons.append("没有可引用新闻，因此只能给出低置信度。")
    elif prices and ranked_articles:
        reasons.append("同时具备行情、新闻、引用、词云和行业映射，可支持关联解释与探索性假设。")
    return reasons


def _coverage_status(citations, prices, ranked_articles) -> str:
    if not citations:
        return "证据不足：没有可引用新闻。"
    if not prices:
        return "证据不足：没有行情数据。"
    if not ranked_articles:
        return "证据偏弱：没有进入推理上下文的新闻。"
    return "证据链完整：行情、新闻、时间窗口、引用、词云和行业映射均可展示。"


def _save_analysis(
    db: Session | None,
    request: AnalyzeRequest,
    start_date: date,
    end_date: date,
    answer: str,
    market_summary: dict[str, Any],
    news_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    evaluation: dict[str, Any],
    risk_warnings: list[str],
    citations: list[CitationOut],
) -> None:
    if db is None:
        return
    try:
        analysis = Analysis(
            ticker=request.ticker.upper(),
            question=request.question,
            start_date=start_date,
            end_date=end_date,
            answer=answer,
            market_summary_json=market_summary,
            news_summary_json=news_summary,
            graph_summary_json=graph_summary,
            evaluation_json=evaluation,
            risk_warnings_json=risk_warnings,
        )
        db.add(analysis)
        db.flush()
        for citation in citations:
            db.add(
                Citation(
                    analysis_id=analysis.id,
                    article_id=None,
                    citation_text=citation.citation_text,
                    url=citation.url,
                    source_domain=citation.source_domain,
                    published_at=citation.published_at,
                )
            )
        db.commit()
    except Exception:
        db.rollback()
