CREATE TABLE IF NOT EXISTS companies (
  id SERIAL PRIMARY KEY,
  ticker VARCHAR(32) NOT NULL,
  company_name VARCHAR(255) NOT NULL,
  exchange VARCHAR(64),
  aliases JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker
ON companies (ticker);

CREATE INDEX IF NOT EXISTS idx_companies_company_name
ON companies (company_name);

CREATE INDEX IF NOT EXISTS idx_companies_aliases
ON companies USING GIN (aliases);

CREATE TABLE IF NOT EXISTS market_prices (
  id SERIAL PRIMARY KEY,
  ticker VARCHAR(32) NOT NULL,
  trade_date DATE NOT NULL,
  open NUMERIC(18, 6) NOT NULL,
  high NUMERIC(18, 6) NOT NULL,
  low NUMERIC(18, 6) NOT NULL,
  close NUMERIC(18, 6) NOT NULL,
  volume INTEGER NOT NULL,
  adjusted_close NUMERIC(18, 6),
  data_source VARCHAR(64) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_market_prices_ticker_date
ON market_prices (ticker, trade_date);

CREATE TABLE IF NOT EXISTS news_articles (
  id SERIAL PRIMARY KEY,
  ticker VARCHAR(32) NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  source_domain VARCHAR(255) NOT NULL,
  published_at TIMESTAMPTZ NOT NULL,
  tone VARCHAR(64),
  language VARCHAR(32),
  short_snippet VARCHAR(300),
  raw_source VARCHAR(64) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_news_articles_ticker_published
ON news_articles (ticker, published_at);

CREATE TABLE IF NOT EXISTS analyses (
  id SERIAL PRIMARY KEY,
  ticker VARCHAR(32) NOT NULL,
  question TEXT NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  answer TEXT NOT NULL,
  market_summary_json JSONB NOT NULL,
  news_summary_json JSONB NOT NULL,
  graph_summary_json JSONB NOT NULL,
  evaluation_json JSONB NOT NULL,
  risk_warnings_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS citations (
  id SERIAL PRIMARY KEY,
  analysis_id INTEGER REFERENCES analyses(id),
  article_id INTEGER REFERENCES news_articles(id),
  citation_text TEXT NOT NULL,
  url TEXT NOT NULL,
  source_domain VARCHAR(255) NOT NULL,
  published_at TIMESTAMPTZ NOT NULL
);
