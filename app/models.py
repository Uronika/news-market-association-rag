from .db import SQLALCHEMY_AVAILABLE


if SQLALCHEMY_AVAILABLE:
    from sqlalchemy import Date, DateTime, ForeignKey, Integer, JSON, Numeric, String, Text, func
    from sqlalchemy.orm import Mapped, mapped_column, relationship

    from .db import Base

    class Company(Base):
        __tablename__ = "companies"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        ticker: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
        company_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
        exchange: Mapped[str | None] = mapped_column(String(64))
        aliases: Mapped[list[str] | None] = mapped_column(JSON)
        created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())

    class MarketPrice(Base):
        __tablename__ = "market_prices"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        ticker: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
        trade_date: Mapped[object] = mapped_column(Date, index=True, nullable=False)
        open: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
        high: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
        low: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
        close: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
        volume: Mapped[int] = mapped_column(Integer, nullable=False)
        adjusted_close: Mapped[float | None] = mapped_column(Numeric(18, 6))
        data_source: Mapped[str] = mapped_column(String(64), nullable=False)
        created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())

    class NewsArticle(Base):
        __tablename__ = "news_articles"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        ticker: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
        title: Mapped[str] = mapped_column(Text, nullable=False)
        url: Mapped[str] = mapped_column(Text, nullable=False)
        source_domain: Mapped[str] = mapped_column(String(255), nullable=False)
        published_at: Mapped[object] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
        tone: Mapped[str | None] = mapped_column(String(64))
        language: Mapped[str | None] = mapped_column(String(32))
        short_snippet: Mapped[str | None] = mapped_column(String(300))
        raw_source: Mapped[str] = mapped_column(String(64), nullable=False)
        created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())

    class Analysis(Base):
        __tablename__ = "analyses"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        ticker: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
        question: Mapped[str] = mapped_column(Text, nullable=False)
        start_date: Mapped[object] = mapped_column(Date, nullable=False)
        end_date: Mapped[object] = mapped_column(Date, nullable=False)
        answer: Mapped[str] = mapped_column(Text, nullable=False)
        market_summary_json: Mapped[dict] = mapped_column(JSON, nullable=False)
        news_summary_json: Mapped[dict] = mapped_column(JSON, nullable=False)
        graph_summary_json: Mapped[dict] = mapped_column(JSON, nullable=False)
        evaluation_json: Mapped[dict] = mapped_column(JSON, nullable=False)
        risk_warnings_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
        created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())
        citations: Mapped[list["Citation"]] = relationship(back_populates="analysis")

    class Citation(Base):
        __tablename__ = "citations"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        analysis_id: Mapped[int | None] = mapped_column(ForeignKey("analyses.id"))
        article_id: Mapped[int | None] = mapped_column(ForeignKey("news_articles.id"))
        citation_text: Mapped[str] = mapped_column(Text, nullable=False)
        url: Mapped[str] = mapped_column(Text, nullable=False)
        source_domain: Mapped[str] = mapped_column(String(255), nullable=False)
        published_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False)
        analysis: Mapped[Analysis] = relationship(back_populates="citations")
else:
    class _UnavailableModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    Company = MarketPrice = NewsArticle = Analysis = Citation = _UnavailableModel
