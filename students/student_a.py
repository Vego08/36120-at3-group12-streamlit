"""Student A: Asset Performance Index (API), XGBoost regression.

Owner: Student A. Replace the starter visualisation and add your own content.
Required outputs (brief): Performance Score, Feature Importance.
"""

from __future__ import annotations

import streamlit as st

from common.api_client import render_feature_importance, render_prediction
from common.config import STUDENTS, selected_token
from common.data import load_history

STUDENT = "A"
TITLE = STUDENTS[STUDENT]["target"]


def home_visual(token: str) -> None:
    """Starter home-page visualisation: 20-day return (replace with your own)."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    return20 = history["close"].pct_change(20).rename("20-day return").dropna()
    st.line_chart(return20.tail(180), height=220)


def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {token}")
    st.caption("XGBoost regression, 7 days ahead")
    render_prediction(STUDENT, token, "Performance Score")
    st.subheader("Feature importance")
    render_feature_importance(STUDENT)
    # TODO (Student A): add explanation, history of the index, features from /features, etc.
