from collections import Counter
from typing import Any

try:
    import networkx as nx
except ImportError:
    nx = None


def build_graph_summary(
    ticker: str,
    company_name: str,
    articles: list[dict[str, Any]],
    market_summary: dict[str, Any],
    reasoning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = nx.Graph() if nx is not None else None
    node_ids: set[str] = set()
    visual_nodes: dict[str, dict[str, Any]] = {}
    visual_links: list[dict[str, Any]] = []
    edge_count = 0
    company_node = f"company:{company_name}"
    ticker_node = f"ticker:{ticker}"
    _add_node(graph, node_ids, visual_nodes, company_node, type="Company", label=company_name)
    _add_node(graph, node_ids, visual_nodes, ticker_node, type="Ticker", label=ticker)
    edge_count += _add_edge(graph, visual_links, company_node, ticker_node, type="COMPANY_HAS_TICKER")

    topic_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    event_windows = [move["trade_date"] for move in market_summary.get("abnormal_moves", [])]

    for idx, article in enumerate(articles):
        article_node = f"article:{idx}"
        source = article.get("source_domain") or "unknown"
        topic = _topic_from_text(article.get("title", ""), article.get("short_snippet", ""))
        source_node = f"source:{source}"
        topic_node = f"topic:{topic}"
        _add_node(graph, node_ids, visual_nodes, article_node, type="NewsArticle", label=article.get("title", ""))
        _add_node(graph, node_ids, visual_nodes, source_node, type="SourceDomain", label=source)
        _add_node(graph, node_ids, visual_nodes, topic_node, type="Topic", label=topic)
        edge_count += _add_edge(graph, visual_links, article_node, company_node, type="ARTICLE_MENTIONS_COMPANY")
        edge_count += _add_edge(graph, visual_links, article_node, source_node, type="ARTICLE_FROM_SOURCE")
        edge_count += _add_edge(graph, visual_links, article_node, topic_node, type="ARTICLE_HAS_TOPIC")
        source_counter[source] += 1
        topic_counter[topic] += 1

        for event_date in event_windows:
            window_node = f"window:{event_date}"
            move_node = f"move:{event_date}"
            _add_node(graph, node_ids, visual_nodes, window_node, type="EventWindow", label=event_date)
            _add_node(graph, node_ids, visual_nodes, move_node, type="MarketMove", label=event_date)
            edge_count += _add_edge(graph, visual_links, window_node, move_node, type="EVENT_WINDOW_HAS_MARKET_MOVE")
            edge_count += _add_edge(graph, visual_links, article_node, window_node, type="ARTICLE_IN_EVENT_WINDOW")

    if reasoning_context:
        for company in reasoning_context.get("mentioned_companies", [])[:8]:
            company_id = f"related_company:{company['ticker']}"
            _add_node(graph, node_ids, visual_nodes, company_id, type="RelatedCompany", label=f"{company['company_name']}({company['ticker']})")
            edge_count += _add_edge(graph, visual_links, company_node, company_id, type="COMPANY_RELATED_TO_COMPANY")

        for industry in reasoning_context.get("industry_impacts", [])[:8]:
            industry_id = f"industry:{industry['id']}"
            _add_node(graph, node_ids, visual_nodes, industry_id, type="Industry", label=industry["industry"])
            edge_count += _add_edge(graph, visual_links, company_node, industry_id, type="COMPANY_LINKED_TO_INDUSTRY")
            for keyword in industry.get("matched_keywords", [])[:5]:
                keyword_id = f"keyword:{keyword}"
                _add_node(graph, node_ids, visual_nodes, keyword_id, type="Keyword", label=keyword)
                edge_count += _add_edge(graph, visual_links, keyword_id, industry_id, type="KEYWORD_SIGNALS_INDUSTRY")

        for item in reasoning_context.get("word_cloud", [])[:15]:
            keyword = item["text"]
            keyword_id = f"word:{keyword}"
            _add_node(graph, node_ids, visual_nodes, keyword_id, type="Keyword", label=keyword, weight=item["weight"])
            edge_count += _add_edge(graph, visual_links, keyword_id, company_node, type="WORD_ASSOCIATED_WITH_CONTEXT")

    return {
        "nodes": graph.number_of_nodes() if graph is not None else len(node_ids),
        "edges": graph.number_of_edges() if graph is not None else edge_count,
        "top_sources": source_counter.most_common(5),
        "top_topics": topic_counter.most_common(5),
        "event_windows": event_windows,
        "nodes_data": list(visual_nodes.values()),
        "links": visual_links,
        "reasoning_context": reasoning_context or {},
        "annotations": _graph_annotations(reasoning_context or {}),
    }


def _add_node(
    graph: Any,
    node_ids: set[str],
    visual_nodes: dict[str, dict[str, Any]],
    node_id: str,
    **attrs: Any,
) -> None:
    node_ids.add(node_id)
    visual_nodes[node_id] = {"id": node_id, **attrs}
    if graph is not None:
        graph.add_node(node_id, **attrs)


def _add_edge(graph: Any, visual_links: list[dict[str, Any]], left: str, right: str, **attrs: Any) -> int:
    visual_links.append({"source": left, "target": right, **attrs})
    if graph is not None:
        graph.add_edge(left, right, **attrs)
    return 1


def _topic_from_text(title: str, snippet: str) -> str:
    text = f"{title} {snippet}".lower()
    for keyword in ["earnings", "delivery", "regulatory", "demand", "guidance", "lawsuit"]:
        if keyword in text:
            return keyword
    return "general"


def _graph_annotations(reasoning_context: dict[str, Any]) -> list[dict[str, str]]:
    scope = reasoning_context.get("analysis_scope", "single_company")
    return [
        {
            "title": "图谱用途",
            "text": "该图谱不是静态知识库，而是本次问题的临时证据图：把自动识别出的企业、新闻、关键词、行业、来源和行情异动窗口连接起来，辅助 RAG 生成可解释回答。",
        },
        {
            "title": "节点说明",
            "text": "Company/Ticker 表示本次自动选择的主分析企业；RelatedCompany 表示问题或新闻中出现的其他企业；Industry 和 Keyword 来自新闻词汇与行业规则映射；NewsArticle 和 SourceDomain 对应可引用新闻证据。",
        },
        {
            "title": "边说明",
            "text": "边表示探索性关系：新闻提及企业、新闻来自来源、新闻包含主题、关键词指向行业、企业连接行业、新闻靠近行情异动窗口。边只表示关联线索，不表示确定因果。",
        },
        {
            "title": "当前问题范围",
            "text": f"本次问题被识别为 {scope}；因此图谱会优先展示与该范围相关的企业关系、行业影响和词云关键词。",
        },
    ]
