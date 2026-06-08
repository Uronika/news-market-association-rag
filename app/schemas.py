from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    ticker: str = Field(default="SPY", min_length=1, max_length=32)
    company_name: str = Field(default="Broad Market", min_length=1, max_length=255)
    start_date: date | None = None
    end_date: date | None = None
    question: str = Field(..., min_length=1)
    top_k_news: int | None = Field(default=None, ge=100)
    aliases: list[str] = Field(default_factory=list)
    deepseek_api_key: str | None = Field(default=None, exclude=True)
    save_deepseek_api_key: bool = False


class CitationOut(BaseModel):
    citation_text: str
    url: str
    source_domain: str
    published_at: datetime


class AnalyzeResponse(BaseModel):
    answer: str
    claim_level: str = "association_only"
    retrieval_trace: list[dict[str, Any]]
    market_summary: dict[str, Any]
    news_summary: dict[str, Any]
    graph_summary: dict[str, Any]
    citations: list[CitationOut]
    evaluation: dict[str, Any]
    risk_warnings: list[str]


class CompanySearchResult(BaseModel):
    ticker: str
    company_name: str
    exchange: str | None = None
    aliases: list[str] = Field(default_factory=list)
    source: str = "local"
