from datetime import date

import anyio
import pytest

from app.config import Settings
from app.embed import NoopEmbedder, get_embedder
from app.sources import (
    AlphaVantageMarketSource,
    GDELTNewsSource,
    MockMarketSource,
    MockNewsSource,
    SourceError,
    StooqMarketSource,
    YahooFinanceMarketSource,
    YahooFinanceNewsSource,
    _date_chunks,
    _dedupe_articles,
    _news_query_terms,
    normalize_stooq_symbol,
)


def test_mock_sources_remain_available_as_test_fixtures():
    async def run():
        news = await MockNewsSource().search_news("Tesla", "TSLA", [], date(2024, 1, 1), date(2024, 1, 31), 2)
        prices = await MockMarketSource().get_prices("TSLA", date(2024, 1, 1), date(2024, 1, 10))
        return news, prices

    news, prices = anyio.run(run)

    assert len(news) == 2
    assert {"title", "url", "source_domain", "published_at", "short_snippet", "raw_source"} <= set(news[0])
    assert prices
    assert prices[0]["data_source"] == "mock"


def test_stooq_symbol_adds_us_suffix_by_default():
    assert normalize_stooq_symbol("TSLA", ".us") == "tsla.us"
    assert normalize_stooq_symbol("0700.hk", ".us") == "0700.hk"


def test_stooq_csv_is_parsed(monkeypatch):
    class FakeResponse:
        text = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,102,99,101,12345\n"

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params):
            assert params["s"] == "tsla.us"
            return FakeResponse()

    monkeypatch.setattr("app.sources.httpx.AsyncClient", lambda timeout: FakeClient())
    settings = Settings()
    source = StooqMarketSource(settings)

    prices = anyio.run(source.get_prices, "TSLA", date(2024, 1, 1), date(2024, 1, 31))

    assert prices == [
        {
            "ticker": "TSLA",
            "trade_date": "2024-01-02",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 12345,
            "adjusted_close": None,
            "data_source": "stooq",
        }
    ]


def test_yahoo_finance_market_response_is_parsed(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1704153600],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [100.0],
                                        "high": [102.0],
                                        "low": [99.0],
                                        "close": [101.0],
                                        "volume": [12345],
                                    }
                                ],
                                "adjclose": [{"adjclose": [101.0]}],
                            },
                        }
                    ],
                    "error": None,
                }
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params):
            assert url.endswith("/TSLA")
            assert params["interval"] == "1d"
            return FakeResponse()

    monkeypatch.setattr("app.sources.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())
    prices = anyio.run(YahooFinanceMarketSource(Settings()).get_prices, "TSLA", date(2024, 1, 1), date(2024, 1, 31))

    assert prices[0]["ticker"] == "TSLA"
    assert prices[0]["data_source"] == "yahoo_finance"
    assert prices[0]["close"] == 101.0


def test_yahoo_finance_news_response_is_parsed(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "news": [
                    {
                        "title": "Tesla real recent news",
                        "publisher": "Yahoo Finance",
                        "link": "https://finance.yahoo.com/news/tesla-real-news",
                        "providerPublishTime": 1704153600,
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params):
            assert params["newsCount"] == 5
            return FakeResponse()

    monkeypatch.setattr("app.sources.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())
    articles = anyio.run(
        YahooFinanceNewsSource(Settings()).search_news,
        "Tesla",
        "TSLA",
        [],
        date(2024, 1, 1),
        date(2024, 1, 31),
        5,
    )

    assert articles[0]["raw_source"] == "yahoo_finance"
    assert articles[0]["title"] == "Tesla real recent news"


def test_yahoo_news_query_terms_filter_chinese_generic_words():
    terms = _news_query_terms("Broad Market", "SPY", ["最近", "AI", "EV", "interest", "rate", "行业"])

    assert "最近" not in terms
    assert "行业" not in terms
    assert "AI" in terms
    assert "EV" in terms


def test_gdelt_chunks_and_deduplicates_articles():
    chunks = list(_date_chunks(date(2024, 1, 1), date(2024, 2, 5), 30))
    articles = _dedupe_articles(
        [
            {"url": "https://a.example/1", "title": "A"},
            {"url": "https://a.example/1", "title": "A duplicate"},
            {"url": "", "title": "Same title"},
            {"url": "", "title": "Same title"},
        ]
    )

    assert chunks == [(date(2024, 1, 1), date(2024, 1, 30)), (date(2024, 1, 31), date(2024, 2, 5))]
    assert len(articles) == 2


def test_gdelt_chunk_response_is_parsed():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "articles": [
                    {
                        "title": "Tesla real news",
                        "url": "https://news.example/tesla",
                        "domain": "news.example",
                        "seendate": "20240102T010000Z",
                        "language": "English",
                        "tone": "0.1",
                    }
                ]
            }

    class FakeClient:
        async def get(self, url, params):
            assert params["mode"] == "artlist"
            return FakeResponse()

    settings = Settings()
    source = GDELTNewsSource(settings)

    articles = anyio.run(
        source._search_chunk,
        FakeClient(),
        "Tesla",
        "TSLA",
        [],
        date(2024, 1, 1),
        date(2024, 1, 31),
        10,
    )

    assert articles[0]["raw_source"] == "gdelt"
    assert articles[0]["source_domain"] == "news.example"


def test_external_market_source_without_key_returns_clear_error():
    settings = Settings()
    settings.alpha_vantage_api_key = "replace-if-needed"
    source = AlphaVantageMarketSource(settings)

    async def run():
        await source.get_prices("TSLA", date(2024, 1, 1), date(2024, 1, 31))

    with pytest.raises(SourceError, match="API key is not configured"):
        anyio.run(run)


def test_env_example_has_no_real_secret():
    text = open(".env.example", encoding="utf-8").read()

    assert "replace-if-needed" in text
    assert "sk-" not in text


def test_noop_embedder_runs_without_external_model():
    settings = Settings()
    settings.embed_provider = "none"

    embedder = get_embedder(settings)

    assert isinstance(embedder, NoopEmbedder)
    assert embedder.embed_texts(["hello"]) == [[]]
