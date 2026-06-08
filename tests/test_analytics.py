from app.analytics import summarize_market


def test_daily_return_and_abnormal_move_are_calculated():
    prices = [
        {"trade_date": "2024-01-01", "close": 100, "volume": 100, "ticker": "T", "open": 100, "high": 101, "low": 99, "adjusted_close": None, "data_source": "mock"},
        {"trade_date": "2024-01-02", "close": 110, "volume": 120, "ticker": "T", "open": 100, "high": 111, "low": 99, "adjusted_close": None, "data_source": "mock"},
        {"trade_date": "2024-01-03", "close": 96.8, "volume": 140, "ticker": "T", "open": 110, "high": 111, "low": 96, "adjusted_close": None, "data_source": "mock"},
    ]

    summary = summarize_market(prices, top_n=1)

    assert summary["market_data_days"] == 3
    assert summary["daily_returns"][0]["daily_return"] == 0.1
    assert len(summary["price_series"]) == 3
    assert summary["price_series"][0]["close"] == 100.0
    assert summary["price_series"][1]["daily_return"] == 0.1
    assert summary["abnormal_moves"][0]["trade_date"] == "2024-01-03"
    assert summary["abnormal_moves"][0]["daily_return"] == -0.12


def test_empty_market_summary_is_explicit():
    summary = summarize_market([])

    assert summary["market_data_days"] == 0
    assert summary["abnormal_moves"] == []
    assert summary["price_series"] == []
