"""Cryptocurrency Financial Intelligence Platform: Streamlit entry point.

Run locally from the repository root:
    streamlit run app/main.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Make the repository root importable (common/ and students/ live next to app/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from common.api_client import service_status_table  # noqa: E402
from common.config import DISCLAIMER, STUDENTS, TOKEN_NAMES, TOKENS  # noqa: E402
from common.data import load_history  # noqa: E402
from students import student_a, student_b, student_c, student_d  # noqa: E402

STUDENT_MODULES = {"A": student_a, "B": student_b, "C": student_c, "D": student_d}

st.set_page_config(page_title="Crypto Financial Intelligence", page_icon="📈", layout="wide")


def sidebar() -> None:
    st.sidebar.title("Crypto Intelligence")
    st.sidebar.selectbox(
        "Cryptocurrency", TOKENS, key="token",
        format_func=lambda t: f"{t} ({TOKEN_NAMES[t]})",
    )
    st.sidebar.caption("Predictions are made 7 days ahead by four independent ML services.")


def home() -> None:
    token = st.session_state.get("token", TOKENS[0])
    st.title(f"{TOKEN_NAMES[token]} ({token})")
    st.caption(DISCLAIMER)

    history = load_history(token)
    if history is None:
        st.warning("Historical market data is temporarily unavailable from Kraken.")
    else:
        last, prev = history["close"].iloc[-1], history["close"].iloc[-2]
        c1, c2, c3 = st.columns(3)
        c1.metric("Last close (USD)", f"{last:,.4f}" if last < 10 else f"{last:,.2f}",
                  f"{(last / prev - 1):+.2%}")
        c2.metric("30-day change", f"{(last / history['close'].iloc[-31] - 1):+.2%}")
        c3.metric("Data up to", history.index[-1].strftime("%Y-%m-%d"))
        st.line_chart(history["close"].tail(365).rename("Close (USD)"), height=280)

    st.subheader("Market indicators")
    cols = st.columns(2)
    for i, module in enumerate(STUDENT_MODULES.values()):
        with cols[i % 2]:
            try:
                module.home_visual(token)
            except Exception as exc:  # one broken visual never takes down the page
                st.error(f"This visualisation failed to load: {exc}")

    with st.expander("Service status"):
        with st.spinner("Checking the four prediction services..."):
            st.dataframe(service_status_table(), hide_index=True, width="stretch")


pages = [st.Page(home, title="Home", icon="🏠", default=True, url_path="home")]
for key, module in STUDENT_MODULES.items():
    pages.append(st.Page(module.render, title=STUDENTS[key]["target"], url_path=f"student-{key.lower()}"))

sidebar()
st.navigation(pages).run()
