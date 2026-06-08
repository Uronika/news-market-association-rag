from __future__ import annotations

import re
from collections import Counter
from typing import Any


INDUSTRIES: dict[str, dict[str, Any]] = {
    "electric_vehicle": {
        "label": "新能源汽车与自动驾驶",
        "keywords": ["ev", "electric", "vehicle", "tesla", "battery", "autonomous", "charging", "delivery"],
        "companies": ["Tesla(TSLA)", "BYD(002594.SZ)", "Rivian(RIVN)", "NIO(NIO)"],
    },
    "semiconductor": {
        "label": "半导体与 AI 算力",
        "keywords": ["chip", "semiconductor", "ai", "gpu", "nvidia", "tsmc", "supply", "data center"],
        "companies": ["NVIDIA(NVDA)", "AMD(AMD)", "TSMC(TSM)", "Intel(INTC)"],
    },
    "cloud_software": {
        "label": "云计算与软件平台",
        "keywords": ["cloud", "software", "ai", "microsoft", "google", "amazon", "openai", "enterprise"],
        "companies": ["Microsoft(MSFT)", "Alphabet(GOOGL)", "Amazon(AMZN)", "Oracle(ORCL)"],
    },
    "consumer_internet": {
        "label": "消费互联网与广告",
        "keywords": ["advertising", "social", "consumer", "app", "meta", "search", "streaming", "subscription"],
        "companies": ["Meta(META)", "Alphabet(GOOGL)", "Netflix(NFLX)", "Snap(SNAP)"],
    },
    "energy_materials": {
        "label": "能源、材料与供应链",
        "keywords": ["oil", "energy", "lithium", "battery", "mining", "commodity", "supply chain"],
        "companies": ["Exxon Mobil(XOM)", "Chevron(CVX)", "Albemarle(ALB)", "Freeport-McMoRan(FCX)"],
    },
    "financial_services": {
        "label": "金融服务与利率敏感资产",
        "keywords": ["rate", "fed", "bank", "credit", "loan", "inflation", "yield", "financial"],
        "companies": ["JPMorgan(JPM)", "Bank of America(BAC)", "Visa(V)", "Berkshire Hathaway(BRK.B)"],
    },
}

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "after", "before",
    "about", "amid", "will", "are", "was", "were", "has", "have", "had", "its", "their",
    "news", "stock", "stocks", "market", "markets", "company", "companies", "inc", "corp",
    "recent", "latest", "says", "said", "new", "why", "what", "how", "between", "among",
}


def build_reasoning_context(
    question: str,
    articles: list[dict[str, Any]],
    primary_ticker: str,
    primary_company: str,
) -> dict[str, Any]:
    words = _tokenize(" ".join(_article_text(article) for article in articles))
    word_counter = Counter(words)
    word_cloud = [{"text": word, "weight": count} for word, count in word_counter.most_common(40)]
    scope = _classify_scope(question)
    mentioned_companies = _detect_companies(question, articles, primary_ticker, primary_company)
    industry_impacts = _score_industries(word_counter, question)
    company_relationships = _company_relationships(mentioned_companies, industry_impacts)
    causal_hypotheses = _causal_hypotheses(scope, word_cloud, industry_impacts, mentioned_companies)

    return {
        "analysis_scope": scope,
        "reasoning_mode": "exploratory_hypothesis",
        "primary_entity": {"ticker": primary_ticker.upper(), "company_name": primary_company},
        "mentioned_companies": mentioned_companies,
        "industry_impacts": industry_impacts,
        "company_relationships": company_relationships,
        "causal_hypotheses": causal_hypotheses,
        "word_cloud": word_cloud,
        "reasoning_warning": (
            "以下为从新闻词汇、行业映射和企业关系中生成的探索性推理路径，"
            "只能作为关联假设，不能视为已证明的因果结论。"
        ),
    }


def build_reasoning_addendum(context: dict[str, Any]) -> str:
    scope = context.get("analysis_scope", "single_company")
    lines = [
        "",
        "【扩展关联推理】",
        f"分析范围：{_scope_label(scope)}。",
        context["reasoning_warning"],
    ]
    hypotheses = context.get("causal_hypotheses", [])[:4]
    if hypotheses:
        lines.append("可能的相互关系/因果路径假设：")
        lines.extend(f"- {item}" for item in hypotheses)
    industries = context.get("industry_impacts", [])[:5]
    if industries:
        lines.append("可能受影响行业及企业样例：")
        for item in industries:
            companies = "、".join(item["example_companies"][:4])
            lines.append(f"- {item['industry']}：关键词 {', '.join(item['matched_keywords'][:5])}；样例企业：{companies}")
    relationships = context.get("company_relationships", [])[:4]
    if relationships:
        lines.append("企业间关系线索：")
        lines.extend(f"- {item}" for item in relationships)
    return "\n".join(lines)


def _classify_scope(question: str) -> str:
    lower = question.lower()
    if any(token in question for token in ["某个新闻", "这条新闻", "具体新闻", "延伸"]):
        return "news_industry_impact"
    if any(token in question for token in ["各个行业", "所有行业", "行业影响", "产业影响", "某段时间"]):
        return "industry_period_impact"
    if any(token in question for token in ["多企业", "多个企业", "多家公司", "企业和新闻"]):
        return "multi_company_news"
    if any(token in question for token in ["企业与企业", "公司关系", "竞争", "供应链", "合作", "上下游"]):
        return "company_relationship"
    if any(token in lower for token in ["vs", "versus"]) or "和" in question and "关系" in question:
        return "company_relationship"
    return "single_company"


def _score_industries(word_counter: Counter[str], question: str) -> list[dict[str, Any]]:
    lower_question = question.lower()
    results = []
    for industry_id, config in INDUSTRIES.items():
        matched = []
        score = 0
        for keyword in config["keywords"]:
            parts = _tokenize(keyword)
            keyword_score = sum(word_counter.get(part, 0) for part in parts)
            if keyword in lower_question:
                keyword_score += 2
            if keyword_score:
                matched.append(keyword)
                score += keyword_score
        if score:
            results.append(
                {
                    "id": industry_id,
                    "industry": config["label"],
                    "score": score,
                    "matched_keywords": matched,
                    "example_companies": config["companies"],
                    "hypothesized_impact": _impact_sentence(config["label"], matched),
                }
            )
    return sorted(results, key=lambda item: item["score"], reverse=True)


def _detect_companies(
    question: str,
    articles: list[dict[str, Any]],
    primary_ticker: str,
    primary_company: str,
) -> list[dict[str, str]]:
    text = f"{question} " + " ".join(_article_text(article) for article in articles[:30])
    candidates = {
        primary_ticker.upper(): primary_company,
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "AMZN": "Amazon",
        "META": "Meta",
        "AMD": "AMD",
        "INTC": "Intel",
        "RIVN": "Rivian",
        "NIO": "NIO",
        "JPM": "JPMorgan",
    }
    found = []
    lower_text = text.lower()
    for ticker, name in candidates.items():
        if ticker.lower() in lower_text or name.lower() in lower_text:
            found.append({"ticker": ticker, "company_name": name})
    return _dedupe_companies(found)


def _company_relationships(companies: list[dict[str, str]], industries: list[dict[str, Any]]) -> list[str]:
    if len(companies) < 2:
        if industries:
            return [f"{companies[0]['company_name'] if companies else '目标企业'}可能通过{industries[0]['industry']}主题与同业或上下游企业形成间接关联。"]
        return []
    names = [item["company_name"] for item in companies[:4]]
    relationships = []
    for left, right in zip(names, names[1:]):
        relationships.append(f"{left} 与 {right} 可从竞争、供应链、AI/平台生态或投资者风险偏好联动角度建立关系假设。")
    return relationships


def _causal_hypotheses(
    scope: str,
    word_cloud: list[dict[str, Any]],
    industries: list[dict[str, Any]],
    companies: list[dict[str, str]],
) -> list[str]:
    keywords = [item["text"] for item in word_cloud[:8]]
    industry_names = [item["industry"] for item in industries[:3]]
    company_names = [item["company_name"] for item in companies[:4]]
    hypotheses = []
    if keywords:
        hypotheses.append(f"高频词 {', '.join(keywords[:5])} 可能代表市场正在关注的外部变量，可作为解释股价或行业情绪变化的候选线索。")
    if industry_names:
        hypotheses.append(f"这些词汇首先映射到 {'、'.join(industry_names)}，因此影响可能不是单公司孤立事件，而是经由产业链或风险偏好扩散。")
    if len(company_names) >= 2:
        hypotheses.append(f"{'、'.join(company_names)} 之间可能存在竞争、供应链、客户需求或资金风格迁移关系，新闻可能通过这些关系产生联动解释。")
    if scope in {"industry_period_impact", "news_industry_impact"}:
        hypotheses.append("若问题关注某段时间或某条新闻的外溢影响，应优先观察跨行业共同关键词，而不是只看单一 ticker 的价格异动。")
    hypotheses.append("以上路径属于激进关联推理：它把新闻字词与外部宏观、产业链和投资者情绪变量连接起来，但仍需更多证据验证。")
    return hypotheses


def _impact_sentence(industry: str, keywords: list[str]) -> str:
    if not keywords:
        return f"{industry}可能受到新闻情绪或风险偏好变化的间接影响。"
    return f"{industry}可能通过 {', '.join(keywords[:4])} 等关键词与新闻事件建立关联。"


def _article_text(article: dict[str, Any]) -> str:
    return f"{article.get('title', '')} {article.get('short_snippet', '')}"


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
        if token not in STOP_WORDS and len(token) <= 24
    ]


def _scope_label(scope: str) -> str:
    return {
        "single_company": "单企业新闻-行情关联",
        "company_relationship": "企业与企业关系",
        "multi_company_news": "多企业和新闻间关系",
        "industry_period_impact": "某段时间新闻对行业影响",
        "news_industry_impact": "具体新闻及延伸对行业影响",
    }.get(scope, scope)


def _dedupe_companies(companies: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    results = []
    for company in companies:
        key = company["ticker"].upper()
        if key in seen:
            continue
        seen.add(key)
        results.append(company)
    return results
