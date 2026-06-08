from functools import lru_cache
import os

from dotenv import load_dotenv


load_dotenv(override=True)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        load_dotenv(override=True)

        self.app_env: str = os.getenv("APP_ENV", "development")
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/news_market_rag",
        )
        self.use_mock_data: bool = _bool_env("USE_MOCK_DATA", False)

        self.news_provider: str = os.getenv("NEWS_PROVIDER", "yahoo_finance")
        self.enable_gdelt: bool = _bool_env("ENABLE_GDELT", True)
        self.gdelt_base_url: str = os.getenv(
            "GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
        )
        self.yahoo_finance_search_url: str = os.getenv(
            "YAHOO_FINANCE_SEARCH_URL", "https://query1.finance.yahoo.com/v1/finance/search"
        )
        self.yahoo_finance_chart_url: str = os.getenv(
            "YAHOO_FINANCE_CHART_URL", "https://query1.finance.yahoo.com/v8/finance/chart"
        )

        self.market_provider: str = os.getenv("MARKET_PROVIDER", "yahoo_finance")
        self.alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.alpha_vantage_base_url: str = os.getenv(
            "ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"
        )
        self.twelve_data_api_key: str = os.getenv("TWELVE_DATA_API_KEY", "")
        self.twelve_data_base_url: str = os.getenv(
            "TWELVE_DATA_BASE_URL", "https://api.twelvedata.com"
        )
        self.nasdaq_data_link_api_key: str = os.getenv("NASDAQ_DATA_LINK_API_KEY", "")
        self.nasdaq_data_link_base_url: str = os.getenv(
            "NASDAQ_DATA_LINK_BASE_URL", "https://data.nasdaq.com/api/v3"
        )
        self.stooq_base_url: str = os.getenv("STOOQ_BASE_URL", "https://stooq.com/q/d/l/")
        self.stooq_symbol_suffix: str = os.getenv("STOOQ_SYMBOL_SUFFIX", ".us")

        self.llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
        self.deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.deepseek_request_timeout_seconds: float = _float_env("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", 90.0)
        self.proxy_url: str = os.getenv("PROXY_URL", "")

        self.embed_provider: str = os.getenv("EMBED_PROVIDER", "none")
        self.embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-zh-v1.5")

        self.external_request_timeout_seconds: float = _float_env("EXTERNAL_REQUEST_TIMEOUT_SECONDS", 8.0)
        self.yahoo_news_query_limit: int = _int_env("YAHOO_NEWS_QUERY_LIMIT", 3)
        self.related_market_entity_limit: int = _int_env("RELATED_MARKET_ENTITY_LIMIT", 3)
        self.analysis_total_timeout_seconds: float = _float_env("ANALYSIS_TOTAL_TIMEOUT_SECONDS", 0.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
