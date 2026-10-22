"""Student D: Trading Opportunity Rating (TOR), LightGBM multiclass classification.

Owner: Student D.
Required outputs (brief): Recommendation, Prediction Confidence Score, Feature Importance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common.api_client import render_feature_importance, render_prediction
from common.config import DISCLAIMER, STUDENTS, selected_token
from common.data import load_history

STUDENT = "D"
TITLE = STUDENTS[STUDENT]["target"]


def home_visual(token: str) -> None:
    """Daily trading volume compared with its 20-day average (RelativeVolume context)."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    volume = history["volume"]
    chart = pd.DataFrame({"Daily volume": volume, "20-day average": volume.rolling(20).mean()})
    st.line_chart(chart.tail(120), height=220)


def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {token}")
    st.caption("LightGBM multiclass classification, 7 days ahead")
    render_prediction(STUDENT, token, "Recommendation")
    st.caption(DISCLAIMER)
    st.subheader("Feature importance")
    render_feature_importance(STUDENT)
    # TODO (Student D): class probabilities, per-prediction explanation, volume features.
