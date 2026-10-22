"""Historical market data for charts, retrieved live from the Kraken public API.

Used only for display. Predictions always come from the FastAPI services.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
TOKEN_TO_PAIR = {
    "BTC": "XBTUSD", "ETH": "ETHUSD", "XRP": "XRPUSD", "SOL": "SOLUSD",
    "DOGE": "XDGUSD", "XMR": "XMRUSD", "XLM": "XLMUSD", "TRX": "TRXUSD",
}
COLUMNS = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(token: str) -> pd.DataFrame | None:
    """Completed daily candles (up to 720 days) for one token, or None if Kraken is unavailable."""
    try:
        r = requests.get(KRAKEN_OHLC_URL, params={"pair": TOKEN_TO_PAIR[token], "interval": 1440},
                         timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            return None
        key = next(k for k in payload["result"] if k != "last")
        df = pd.DataFrame(payload["result"][key], columns=COLUMNS)
    except Exception:
        return None

    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return df[df["date"] < today].set_index("date")[["open", "high", "low", "close", "volume"]]
