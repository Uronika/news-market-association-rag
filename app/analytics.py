from typing import Any

import numpy as np
import pandas as pd


def summarize_market(prices: list[dict[str, Any]], top_n: int = 3) -> dict[str, Any]:
    if not prices:
        return {
            "market_data_days": 0,
            "abnormal_moves": [],
            "daily_returns": [],
            "price_series": [],
            "volume_zscores": [],
        }

    df = pd.DataFrame(prices).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df.sort_values("trade_date")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["daily_return"] = df["close"] / df["close"].shift(1) - 1

    volume_std = float(df["volume"].std(ddof=0))
    if volume_std == 0 or np.isnan(volume_std):
        df["volume_zscore"] = 0.0
    else:
        df["volume_zscore"] = (df["volume"] - float(df["volume"].mean())) / volume_std

    ranked = df.dropna(subset=["daily_return"]).copy()
    ranked["abs_return"] = ranked["daily_return"].abs()
    abnormal = ranked.sort_values("abs_return", ascending=False).head(top_n)

    return {
        "market_data_days": int(len(df)),
        "start_date": str(df["trade_date"].iloc[0]),
        "end_date": str(df["trade_date"].iloc[-1]),
        "daily_returns": [
            {
                "trade_date": str(row.trade_date),
                "daily_return": round(float(row.daily_return), 6),
            }
            for row in ranked.itertuples()
        ],
        "price_series": [
            {
                "trade_date": str(row.trade_date),
                "open": round(float(row.open), 6) if "open" in df.columns else round(float(row.close), 6),
                "high": round(float(row.high), 6) if "high" in df.columns else round(float(row.close), 6),
                "low": round(float(row.low), 6) if "low" in df.columns else round(float(row.close), 6),
                "close": round(float(row.close), 6),
                "daily_return": None if pd.isna(row.daily_return) else round(float(row.daily_return), 6),
            }
            for row in df.itertuples()
        ],
        "volume_zscores": [
            {
                "trade_date": str(row.trade_date),
                "volume_zscore": round(float(row.volume_zscore), 6),
            }
            for row in df.itertuples()
        ],
        "abnormal_moves": [
            {
                "trade_date": str(row.trade_date),
                "daily_return": round(float(row.daily_return), 6),
                "abs_return": round(float(row.abs_return), 6),
                "volume_zscore": round(float(row.volume_zscore), 6),
            }
            for row in abnormal.itertuples()
        ],
    }
