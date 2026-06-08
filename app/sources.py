from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import re

from .config import Settings


MIN_REAL_DATA_DATE = date(2010, 1, 1)


class SourceError(RuntimeError):
    pass


class NewsSource:
    async def search_news(
        self,
        company_name: str,
        ticker: str,
        aliases: list[str],
        start_date: date,
        end_date: date,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class MarketSource:
    async def get_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class MockNewsSource(NewsSource):
    async def search_news(self, company_name, ticker, aliases, start_date, end_date, top_k):
        if top_k <= 0:
            return []
        midpoint = start_date + (end_date - start_date) / 2
        topics = [
            "delivery update",
            "earnings discussion",
            "regulatory attention",
            "sector demand",
            "AI chip supply",
            "battery materials",
            "interest rate pressure",
            "cloud platform demand",
            "consumer advertising trend",
            "energy supply chain",
        ]
        articles = []
        for idx in range(1, top_k + 1):
            topic = topics[(idx - 1) % len(topics)]
            published = datetime.combine(
                midpoint + timedelta(days=(idx % 21) - 10),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            articles.append(
                {
                    "ticker": ticker.upper(),
                    "title": f"{company_name} {topic} reported by market media",
                    "url": f"https://example.com/{ticker.lower()}/{idx}",
                    "source_domain": "example.com",
                    "published_at": published,
                    "tone": "neutral",
                    "language": "en",
                    "short_snippet": (
                        f"Mock article about {company_name} and {topic}. "
                        "It is used only as test fixture evidence."
                    )[:300],
                    "raw_source": "mock",
                }
            )
        return articles[:top_k]


class GDELTNewsSource(NewsSource):
    def __init__(self, settings: Settings, chunk_days: int = 30):
        self.settings = settings
        self.chunk_days = chunk_days

    async def search_news(self, company_name, ticker, aliases, start_date, end_date, top_k):
        if top_k <= 0:
            return []
        if start_date < MIN_REAL_DATA_DATE:
            raise SourceError("GDELT real-data mode only accepts start_date >= 2010-01-01")

        all_articles: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=self.settings.external_request_timeout_seconds) as client:
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, self.chunk_days):
                all_articles.extend(
                    await self._search_chunk(
                        client,
                        company_name,
                        ticker,
                        aliases,
                        chunk_start,
                        chunk_end,
                        min(250, max(top_k, 10)),
                    )
                )
                if len(all_articles) >= top_k * 3:
                    break

        deduped = _dedupe_articles(all_articles)
        deduped.sort(key=lambda item: item["published_at"], reverse=True)
        return deduped[:top_k]

    async def _search_chunk(
        self,
        client: httpx.AsyncClient,
        company_name: str,
        ticker: str,
        aliases: list[str],
        start_date: date,
        end_date: date,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_terms = [company_name, ticker, *aliases]
        query = " OR ".join(f'"{term}"' for term in query_terms if term)
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": top_k,
            "startdatetime": start_date.strftime("%Y%m%d000000"),
            "enddatetime": end_date.strftime("%Y%m%d235959"),
            "sort": "hybridrel",
        }
        try:
            resp = await client.get(self.settings.gdelt_base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise SourceError(f"GDELT news request failed for {start_date} to {end_date}: {exc}") from exc

        articles = []
        for item in data.get("articles", [])[:top_k]:
            url = item.get("url") or ""
            title = item.get("title") or "Untitled GDELT article"
            published = _parse_gdelt_datetime(item.get("seendate"))
            articles.append(
                {
                    "ticker": ticker.upper(),
                    "title": title,
                    "url": url,
                    "source_domain": urlparse(url).netloc or item.get("domain") or "unknown",
                    "published_at": published,
                    "tone": str(item.get("tone", "unknown")),
                    "language": item.get("language", "unknown"),
                    "short_snippet": (item.get("snippet") or title)[:300],
                    "raw_source": "gdelt",
                }
            )
        return articles


class YahooFinanceNewsSource(NewsSource):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def search_news(self, company_name, ticker, aliases, start_date, end_date, top_k):
        if top_k <= 0:
            return []
        query_terms = _news_query_terms(company_name, ticker, aliases)[: self.settings.yahoo_news_query_limit]
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        articles = []
        try:
            async with httpx.AsyncClient(timeout=self.settings.external_request_timeout_seconds, headers=headers) as client:
                for query in query_terms:
                    params = {"q": query, "newsCount": min(top_k, 100), "quotesCount": 0}
                    try:
                        resp = await client.get(self.settings.yahoo_finance_search_url, params=params)
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 400:
                            continue
                        raise
                    data = resp.json()
                    articles.extend(_parse_yahoo_news_items(data.get("news", []), ticker))
                    if len(_dedupe_articles(articles)) >= top_k:
                        break
        except Exception as exc:
            raise SourceError(f"Yahoo Finance news request failed: {exc}") from exc

        deduped = _dedupe_articles(articles)
        deduped.sort(key=lambda item: item["published_at"], reverse=True)
        return deduped[:top_k]


class MockMarketSource(MarketSource):
    async def get_prices(self, ticker, start_date, end_date):
        prices = []
        current = start_date
        idx = 0
        close = 100.0
        while current <= end_date:
            if current.weekday() < 5:
                idx += 1
                move = 0.02 if idx == 4 else (-0.035 if idx == 8 else 0.004)
                open_price = close
                close = round(close * (1 + move), 2)
                prices.append(
                    {
                        "ticker": ticker.upper(),
                        "trade_date": current.isoformat(),
                        "open": round(open_price, 2),
                        "high": round(max(open_price, close) * 1.01, 2),
                        "low": round(min(open_price, close) * 0.99, 2),
                        "close": close,
                        "volume": 1_000_000 + idx * 55_000,
                        "adjusted_close": None,
                        "data_source": "mock",
                    }
                )
            current += timedelta(days=1)
        return prices


class AlphaVantageMarketSource(MarketSource):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_prices(self, ticker, start_date, end_date):
        if not self.settings.alpha_vantage_api_key or self.settings.alpha_vantage_api_key == "replace-if-needed":
            raise SourceError("Alpha Vantage API key is not configured")
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "apikey": self.settings.alpha_vantage_api_key,
            "outputsize": "full",
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.external_request_timeout_seconds) as client:
                resp = await client.get(self.settings.alpha_vantage_base_url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise SourceError(f"Alpha Vantage request failed: {exc}") from exc
        series = data.get("Time Series (Daily)")
        if not series:
            raise SourceError("Alpha Vantage response did not contain daily prices")
        return _daily_series_to_prices(ticker, series, start_date, end_date, "alpha_vantage")


class TwelveDataMarketSource(MarketSource):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_prices(self, ticker, start_date, end_date):
        if not self.settings.twelve_data_api_key or self.settings.twelve_data_api_key == "replace-if-needed":
            raise SourceError("Twelve Data API key is not configured")
        params = {
            "symbol": ticker,
            "interval": "1day",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "apikey": self.settings.twelve_data_api_key,
            "outputsize": 5000,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.external_request_timeout_seconds) as client:
                resp = await client.get(f"{self.settings.twelve_data_base_url}/time_series", params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise SourceError(f"Twelve Data request failed: {exc}") from exc
        values = data.get("values")
        if not values:
            raise SourceError("Twelve Data response did not contain prices")
        prices = [_normalize_price(ticker, row, "twelve_data") for row in values]
        return sorted(prices, key=lambda item: item["trade_date"])


class NasdaqDataLinkMarketSource(MarketSource):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_prices(self, ticker, start_date, end_date):
        if not self.settings.nasdaq_data_link_api_key or self.settings.nasdaq_data_link_api_key == "replace-if-needed":
            raise SourceError("Nasdaq Data Link API key is not configured")
        raise SourceError("Nasdaq Data Link adapter requires a dataset code; configure one before use")


class YahooFinanceMarketSource(MarketSource):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_prices(self, ticker, start_date, end_date):
        if start_date < MIN_REAL_DATA_DATE:
            raise SourceError("Yahoo Finance real-data mode only accepts start_date >= 2010-01-01")
        period1 = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
        url = f"{self.settings.yahoo_finance_chart_url.rstrip('/')}/{ticker.upper()}"
        params = {"period1": period1, "period2": period2, "interval": "1d", "events": "history"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.settings.external_request_timeout_seconds, headers=headers) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise SourceError(f"Yahoo Finance chart request failed: {exc}") from exc

        result = (data.get("chart", {}).get("result") or [None])[0]
        if not result:
            error = data.get("chart", {}).get("error")
            raise SourceError(f"Yahoo Finance chart response did not contain prices: {error}")
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
        prices = []
        for idx, ts in enumerate(timestamps):
            open_v = _at(quote.get("open"), idx)
            high_v = _at(quote.get("high"), idx)
            low_v = _at(quote.get("low"), idx)
            close_v = _at(quote.get("close"), idx)
            volume_v = _at(quote.get("volume"), idx)
            if None in {open_v, high_v, low_v, close_v, volume_v}:
                continue
            trade_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            prices.append(
                {
                    "ticker": ticker.upper(),
                    "trade_date": trade_date.isoformat(),
                    "open": float(open_v),
                    "high": float(high_v),
                    "low": float(low_v),
                    "close": float(close_v),
                    "volume": int(volume_v),
                    "adjusted_close": float(_at(adjclose, idx)) if _at(adjclose, idx) is not None else None,
                    "data_source": "yahoo_finance",
                }
            )
        if not prices:
            raise SourceError(f"Yahoo Finance chart response had no parseable prices for {ticker}")
        return sorted(prices, key=lambda item: item["trade_date"])


class StooqMarketSource(MarketSource):
    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_prices(self, ticker, start_date, end_date):
        if start_date < MIN_REAL_DATA_DATE:
            raise SourceError("Stooq real-data mode only accepts start_date >= 2010-01-01")
        symbol = normalize_stooq_symbol(ticker, self.settings.stooq_symbol_suffix)
        params = {
            "s": symbol,
            "d1": start_date.strftime("%Y%m%d"),
            "d2": end_date.strftime("%Y%m%d"),
            "i": "d",
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.external_request_timeout_seconds) as client:
                resp = await client.get(self.settings.stooq_base_url, params=params)
                resp.raise_for_status()
        except Exception as exc:
            raise SourceError(f"Stooq request failed: {exc}") from exc

        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
        if resp.text.lstrip().lower().startswith("<!doctype html") or "requires JavaScript" in resp.text:
            raise SourceError(
                "Stooq returned an HTML browser verification page instead of CSV; "
                "real market data was not retrieved."
            )
        if len(lines) <= 1 or "No data" in resp.text:
            raise SourceError(f"Stooq response did not contain prices for symbol {symbol}")
        prices = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            date_s, open_s, high_s, low_s, close_s, volume_s = parts[:6]
            prices.append(
                {
                    "ticker": ticker.upper(),
                    "trade_date": date_s,
                    "open": float(open_s),
                    "high": float(high_s),
                    "low": float(low_s),
                    "close": float(close_s),
                    "volume": int(float(volume_s)),
                    "adjusted_close": None,
                    "data_source": "stooq",
                }
            )
        if not prices:
            raise SourceError(f"Stooq response had no parseable prices for symbol {symbol}")
        return sorted(prices, key=lambda item: item["trade_date"])


def normalize_stooq_symbol(ticker: str, suffix: str = ".us") -> str:
    raw = ticker.strip().lower()
    if "." in raw:
        return raw
    return f"{raw}{suffix}" if suffix else raw


def _date_chunks(start_date: date, end_date: date, chunk_days: int):
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _dedupe_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for article in articles:
        key = article.get("url") or article.get("title", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def _news_query_terms(company_name: str, ticker: str, aliases: list[str]) -> list[str]:
    raw_terms = [ticker, company_name, *aliases]
    terms = []
    for term in raw_terms:
        clean = str(term or "").strip()
        if not _valid_yahoo_news_query(clean):
            continue
        terms.append(clean)
    combined = " ".join(terms[:6]).strip()
    candidates = [combined, *terms]
    seen = set()
    result = []
    for term in candidates:
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result[:16]


def _valid_yahoo_news_query(term: str) -> bool:
    if not term or len(term) < 2 or len(term) > 80:
        return False
    lower = term.lower()
    blocked = {
        "why", "what", "how", "recent", "latest", "news", "market",
        "最近", "新闻", "可能", "怎样", "怎么", "什么", "影响", "行业", "企业", "关系",
    }
    if lower in blocked:
        return False
    if re.search(r"[\u4e00-\u9fff]", term):
        return False
    return bool(re.search(r"[A-Za-z0-9]", term))


def _parse_yahoo_news_items(items: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    articles = []
    for item in items:
        published_ts = item.get("providerPublishTime")
        published = (
            datetime.fromtimestamp(published_ts, tz=timezone.utc)
            if published_ts
            else datetime.now(timezone.utc)
        )
        url = item.get("link") or ""
        title = item.get("title") or "Untitled Yahoo Finance article"
        articles.append(
            {
                "ticker": ticker.upper(),
                "title": title,
                "url": url,
                "source_domain": urlparse(url).netloc or item.get("publisher") or "finance.yahoo.com",
                "published_at": published,
                "tone": "unknown",
                "language": "en",
                "short_snippet": title[:300],
                "raw_source": "yahoo_finance",
            }
        )
    return articles


def _parse_gdelt_datetime(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _daily_series_to_prices(ticker, series, start_date, end_date, source):
    prices = []
    for date_s, row in series.items():
        trade_date = date.fromisoformat(date_s)
        if start_date <= trade_date <= end_date:
            prices.append(
                {
                    "ticker": ticker.upper(),
                    "trade_date": date_s,
                    "open": float(row["1. open"]),
                    "high": float(row["2. high"]),
                    "low": float(row["3. low"]),
                    "close": float(row["4. close"]),
                    "volume": int(float(row["6. volume"])),
                    "adjusted_close": float(row.get("5. adjusted close") or row["4. close"]),
                    "data_source": source,
                }
            )
    return sorted(prices, key=lambda item: item["trade_date"])


def _normalize_price(ticker, row, source):
    return {
        "ticker": ticker.upper(),
        "trade_date": row["datetime"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(float(row.get("volume") or 0)),
        "adjusted_close": None,
        "data_source": source,
    }


def _at(values, idx):
    if not values or idx >= len(values):
        return None
    return values[idx]


def get_news_source(settings: Settings) -> NewsSource:
    if settings.use_mock_data or settings.news_provider == "mock":
        return MockNewsSource()
    if settings.news_provider == "yahoo_finance":
        return YahooFinanceNewsSource(settings)
    if settings.news_provider == "gdelt" and settings.enable_gdelt:
        return GDELTNewsSource(settings)
    raise SourceError(f"Unsupported or disabled news provider: {settings.news_provider}")


def get_market_source(settings: Settings) -> MarketSource:
    if settings.use_mock_data or settings.market_provider == "mock":
        return MockMarketSource()
    providers = {
        "alpha_vantage": AlphaVantageMarketSource,
        "twelve_data": TwelveDataMarketSource,
        "nasdaq_data_link": NasdaqDataLinkMarketSource,
        "yahoo_finance": YahooFinanceMarketSource,
        "stooq": StooqMarketSource,
    }
    cls = providers.get(settings.market_provider)
    if cls is None:
        raise SourceError(f"Unsupported market provider: {settings.market_provider}")
    return cls(settings)
