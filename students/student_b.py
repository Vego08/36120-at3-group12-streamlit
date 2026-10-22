"""Student B: Market Risk Index (MRI), LightGBM regression.

Owner: Student B. Replace the starter visualisation and add your own content.
Required outputs (brief): Risk Score, Feature Importance.
"""

from __future__ import annotations

import streamlit as st

from common.api_client import render_feature_importance, render_prediction
from common.config import STUDENTS, selected_token
from common.data import load_history

STUDENT = "B"
TITLE = STUDENTS[STUDENT]["target"]


def home_visual(token: str) -> None:
    """Starter home-page visualisation: 20-day rolling volatility (replace with your own)."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    vol = history["close"].pct_change().rolling(20).std().rename("20-day volatility").dropna()
    st.line_chart(vol.tail(180), height=220)


def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {token}")
    st.caption("LightGBM regression, 7 days ahead")
    render_prediction(STUDENT, token, "Risk Score")
    st.subheader("Feature importance")
    render_feature_importance(STUDENT)
    # TODO (Student B): add explanation, risk breakdown, features from /features, etc.
