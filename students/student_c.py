"""Student C: Trend Category (TC), XGBoost multiclass classification.

Owner: Student C. Replace the starter visualisation and add your own content.
Required outputs (brief): Current Trend, Prediction Confidence Score, Feature Importance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common.api_client import render_feature_importance, render_prediction
from common.config import STUDENTS, selected_token
from common.data import load_history

STUDENT = "C"
TITLE = STUDENTS[STUDENT]["target"]


def home_visual(token: str) -> None:
    """Starter home-page visualisation: close price with MA5 and MA20 (replace with your own)."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    close = history["close"]
    chart = pd.DataFrame({"Close": close, "MA5": close.rolling(5).mean(), "MA20": close.rolling(20).mean()})
    st.line_chart(chart.tail(120), height=220)


def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {token}")
    st.caption("XGBoost multiclass classification, 7 days ahead")
    render_prediction(STUDENT, token, "Trend Category")
    st.subheader("Feature importance")
    render_feature_importance(STUDENT)
    # TODO (Student C): add explanation, class probabilities, features from /features, etc.
