from __future__ import annotations

from typing import Any

import httpx

try:
    from sqlalchemy import or_, select
    from sqlalchemy.orm import Session
except ImportError:
    or_ = None
    select = None
    Session = Any

from .config import Settings
from .db import SQLALCHEMY_AVAILABLE
from .models import Company
from .schemas import CompanySearchResult


COMMON_COMPANIES = [
    ("TSLA", "Tesla, Inc.", "NASDAQ", ["Tesla"]),
    ("AAPL", "Apple Inc.", "NASDAQ", ["Apple"]),
    ("MSFT", "Microsoft Corporation", "NASDAQ", ["Microsoft"]),
    ("NVDA", "NVIDIA Corporation", "NASDAQ", ["Nvidia"]),
    ("GOOGL", "Alphabet Inc.", "NASDAQ", ["Google", "Alphabet"]),
    ("AMZN", "Amazon.com, Inc.", "NASDAQ", ["Amazon"]),
    ("META", "Meta Platforms, Inc.", "NASDAQ", ["Meta", "Facebook"]),
]


async def search_companies(query: str, settings: Settings, db: Session | None, limit: int = 8) -> list[CompanySearchResult]:
    query = query.strip()
    if not query:
        return []
    local = _search_local(query, db, limit)
    if len(local) >= limit:
        return local[:limit]

    remote = await _search_yahoo(query, settings, limit)
    _save_companies(db, remote)
    merged = _merge_results([*local, *remote])
    if not merged:
        merged = _search_common(query, limit)
        _save_companies(db, merged)
    return merged[:limit]


def _search_local(query: str, db: Session | None, limit: int) -> list[CompanySearchResult]:
    if db is None or not SQLALCHEMY_AVAILABLE or select is None or or_ is None:
        return _search_common(query, limit)
    pattern = f"%{query}%"
    try:
        rows = db.execute(
            select(Company)
            .where(or_(Company.ticker.ilike(pattern), Company.company_name.ilike(pattern)))
            .limit(limit)
        ).scalars().all()
    except Exception:
        db.rollback()
        return _search_common(query, limit)
    return [
        CompanySearchResult(
            ticker=row.ticker,
            company_name=row.company_name,
            exchange=row.exchange,
            aliases=row.aliases or [],
            source="local",
        )
        for row in rows
    ]


def _search_common(query: str, limit: int) -> list[CompanySearchResult]:
    lower = query.lower()
    results = []
    for ticker, name, exchange, aliases in COMMON_COMPANIES:
        if lower in ticker.lower() or lower in name.lower() or any(lower in alias.lower() for alias in aliases):
            results.append(
                CompanySearchResult(
                    ticker=ticker,
                    company_name=name,
                    exchange=exchange,
                    aliases=aliases,
                    source="built_in",
                )
            )
    return results[:limit]


async def _search_yahoo(query: str, settings: Settings, limit: int) -> list[CompanySearchResult]:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    params = {"q": query, "quotesCount": limit, "newsCount": 0}
    try:
        async with httpx.AsyncClient(timeout=12, headers=headers) as client:
            resp = await client.get(settings.yahoo_finance_search_url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []

    results = []
    for item in data.get("quotes", [])[:limit]:
        symbol = item.get("symbol")
        name = item.get("longname") or item.get("shortname")
        if not symbol or not name:
            continue
        results.append(
            CompanySearchResult(
                ticker=symbol,
                company_name=name,
                exchange=item.get("exchDisp") or item.get("exchange"),
                aliases=[item.get("shortname")] if item.get("shortname") and item.get("shortname") != name else [],
                source="yahoo_finance",
            )
        )
    return results


def _save_companies(db: Session | None, results: list[CompanySearchResult]) -> None:
    if db is None or not SQLALCHEMY_AVAILABLE or select is None:
        return
    try:
        for result in results:
            exists = db.execute(select(Company.id).where(Company.ticker == result.ticker.upper())).first()
            if exists:
                continue
            db.add(
                Company(
                    ticker=result.ticker.upper(),
                    company_name=result.company_name,
                    exchange=result.exchange,
                    aliases=result.aliases,
                )
            )
        db.commit()
    except Exception:
        db.rollback()


def _merge_results(results: list[CompanySearchResult]) -> list[CompanySearchResult]:
    seen = set()
    merged = []
    for result in results:
        key = result.ticker.upper()
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
    return merged
