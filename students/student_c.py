"""Student C: Trend Category (TC), XGBoost multiclass classification.

Owner: Student C (Lavil Saldanha, 26036727).
Required outputs (brief): Current Trend, Prediction Confidence Score, Feature Importance.
All predictions come from the Student C FastAPI service; the TrendScore history chart only applies the
brief's formula to Kraken prices for display, no model runs here.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common.api_client import get_features, get_model_metadata, get_prediction, show_api_error
from common.config import DISCLAIMER, STUDENTS, TOKEN_NAMES, TOKENS, selected_token
from common.data import load_history

STUDENT = "C"
TITLE = STUDENTS[STUDENT]["target"]

LABELS = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
COLOURS = {"Strong Downtrend": "#C62828", "Downtrend": "#EF6C00", "Sideways": "#9E9E9E",
           "Uptrend": "#43A047", "Strong Uptrend": "#1B5E20"}
ICONS = {"Strong Downtrend": "🔴", "Downtrend": "🟠", "Sideways": "⚪", "Uptrend": "🟢", "Strong Uptrend": "🟩"}
MEANING = {
    "Strong Downtrend": "The short-term average is expected to sit well below the 20-day average next week.",
    "Downtrend": "The trend is expected to lean negative next week.",
    "Sideways": "No clear trend is expected next week.",
    "Uptrend": "The trend is expected to lean positive next week.",
    "Strong Uptrend": "The short-term average is expected to sit well above the 20-day average next week.",
}
# TrendScore class boundaries from the brief
BANDS = pd.DataFrame({
    "label": LABELS,
    "low": [-1.0, -0.10, -0.02, 0.02, 0.10],
    "high": [-0.10, -0.02, 0.02, 0.10, 1.0],
})
FEATURE_TEXT = {
    "Return20": "20-day return",
    "Return5": "5-day return",
    "TrendStrength": "Short vs long moving average (MA5 / MA20 - 1)",
    "TrendScore_now": "Today's TrendScore",
    "Volatility20": "20-day volatility",
    "MaxDrawdown": "Drop from the 20-day high",
}
FEATURE_SOURCE = {"Return20": "student_a", "Return5": "student_a", "TrendStrength": "student_b",
                  "TrendScore_now": "derived", "Volatility20": "student_c", "MaxDrawdown": "student_c"}
SOURCE_TEXT = {"student_a": "Student A API", "student_b": "Student B API (MA5, MA20)",
               "student_c": "This service (Kraken)", "derived": "derived (TrendStrength, Return20)"}
# Accuracy of the final model on the held-out test period (25 Jun to 14 Oct 2026), by confidence band
CONFIDENCE_BANDS = pd.DataFrame({
    "Confidence": ["up to 40%", "40 to 50%", "50 to 60%", "60 to 70%", "above 70%"],
    "Predictions": [217, 325, 256, 78, 20],
    "Test accuracy": [0.429, 0.529, 0.578, 0.526, 0.950],
})


def _trend_score(history: pd.DataFrame) -> pd.Series:
    close = history["close"]
    trend_strength = close.rolling(5).mean() / close.rolling(20).mean() - 1
    return (0.7 * trend_strength + 0.3 * close.pct_change(20)).rename("TrendScore")


def _trend_chart(history: pd.DataFrame, days: int, height: int) -> alt.LayerChart:
    """TrendScore over time on top of the five class bands."""
    ts = _trend_score(history).dropna().tail(days)
    df = ts.reset_index()
    pad = 0.02
    lo, hi = min(df["TrendScore"].min() - pad, -0.12), max(df["TrendScore"].max() + pad, 0.12)
    bands = BANDS.assign(low=BANDS["low"].clip(lower=lo), high=BANDS["high"].clip(upper=hi))
    y_scale = alt.Scale(domain=[lo, hi])
    shade = alt.Chart(bands).mark_rect(opacity=0.25).encode(
        y=alt.Y("low:Q", scale=y_scale, title="TrendScore"), y2="high:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=LABELS, range=[COLOURS[l] for l in LABELS]),
                        title=None, legend=alt.Legend(orient="bottom", columns=3, labelLimit=140)),
    )
    line = alt.Chart(df).mark_line(color="#2F80ED", strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("TrendScore:Q", scale=y_scale),
        tooltip=["date:T", alt.Tooltip("TrendScore:Q", format="+.3f")],
    )
    return (shade + line).properties(height=height)


# ---------------------------------------------------------------------------
# Home page visual
# ---------------------------------------------------------------------------
def home_visual(token: str) -> None:
    """TrendScore over the last 120 days, shaded by the five Trend Category bands."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    st.altair_chart(_trend_chart(history, days=120, height=220), width="stretch")


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def _current_trend(data: dict) -> None:
    label = data["prediction"]
    colour = COLOURS.get(label, "#555")
    st.markdown(
        f"""<div style="border-left: 8px solid {colour}; padding: 0.8rem 1.2rem; border-radius: 6px;
        background: rgba(127,127,127,0.08); margin-bottom: 0.8rem;">
        <div style="font-size: 0.9rem; opacity: 0.8;">Trend expected on {data['prediction_date']}
        (based on data up to {data['as_of']})</div>
        <div style="font-size: 2.2rem; font-weight: 700; color: {colour};">{ICONS.get(label, '')} {label}</div>
        <div style="font-size: 0.95rem;">{MEANING.get(label, '')}</div></div>""",
        unsafe_allow_html=True,
    )
    features = data.get("features", {})
    available = sum(v is not None for v in features.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction confidence", f"{data['confidence']:.0%}")
    c2.metric("Forecast horizon", f"{data['forecast_horizon']} days")
    c3.metric("Inputs available", f"{available} of {len(features)}")
    st.progress(min(max(float(data["confidence"]), 0.0), 1.0))
    if data["confidence"] < 0.5:
        st.caption("Confidence is below 50%, so treat this as an uncertain call.")
    if label == "Strong Downtrend":
        st.caption("In testing the model did not identify Strong Downtrend days reliably; check the Market "
                   "Risk Index as well.")


def _probabilities(data: dict) -> None:
    probs = data.get("probabilities", {})
    df = pd.DataFrame({"Trend": LABELS, "Probability": [probs.get(l, 0.0) for l in LABELS]})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Trend:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0, labelLimit=120)),
        y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Trend:N", scale=alt.Scale(domain=LABELS, range=[COLOURS[l] for l in LABELS]),
                        legend=None),
        tooltip=["Trend:N", alt.Tooltip("Probability:Q", format=".1%")],
    ).properties(height=260)
    st.altair_chart(chart, width="stretch")


def _inputs(data: dict) -> None:
    features = data.get("features", {})
    sources = data.get("feature_sources", {})
    rows = []
    for name, value in features.items():
        source = FEATURE_SOURCE.get(name, "")
        rows.append({"Feature": name, "Description": FEATURE_TEXT.get(name, name),
                     "Source": SOURCE_TEXT.get(source, source),
                     "Value": "missing" if value is None else f"{value:+.4f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    missing = [r["Feature"] for r in rows if r["Value"] == "missing"]
    if missing:
        st.info("Some inputs were unavailable and passed to the model as missing values: "
                + ", ".join(missing) + ". The model was trained with simulated outages, but the prediction is "
                "less reliable without them.")
    if sources:
        st.caption("Feature services: " + " · ".join(f"{k.replace('_', ' ').title()}: {v}"
                                                      for k, v in sources.items()))


def _risk_features(token: str) -> None:
    res = get_features(STUDENT, token)
    if not res.ok:
        show_api_error(res, "Trend and risk features")
        return
    f = res.data.get("features", {})

    def fmt(value, spec):
        return "n/a" if value is None else format(value, spec)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("20-day volatility", fmt(f.get("Volatility20"), ".2%"))
    c2.metric("Drop from 20-day high", fmt(f.get("MaxDrawdown"), ".2%"))
    price = f.get("MaxPrice20")
    c3.metric("20-day high (USD)", "n/a" if price is None else (f"{price:,.4f}" if price < 10 else f"{price:,.2f}"))
    c4.metric("Gap risk", fmt(f.get("GapRisk"), ".4%"))
    st.caption(f"As of {res.data.get('as_of', '')}. Gap risk stays near zero because crypto trades around the "
               "clock, so each day opens at almost exactly the previous close.")


def _model_details() -> None:
    res = get_model_metadata(STUDENT)
    if not res.ok:
        show_api_error(res, "Model metadata")
        return
    meta = res.data
    importance = meta.get("feature_importance", {})
    if importance:
        df = pd.DataFrame({"feature": list(importance), "importance": list(importance.values())})
        df["description"] = df["feature"].map(FEATURE_TEXT).fillna(df["feature"])
        chart = alt.Chart(df).mark_bar(color="#4C72B0").encode(
            x=alt.X("importance:Q", title="Share of total split gain", axis=alt.Axis(format="%")),
            y=alt.Y("description:N", sort="-x", title=None, axis=alt.Axis(labelLimit=400)),
            tooltip=["feature:N", alt.Tooltip("importance:Q", format=".1%")],
        ).properties(height=240)
        st.altair_chart(chart, width="stretch")
    metrics = meta.get("test_metrics") or {}
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test macro F1", f"{metrics.get('macro_f1', 0):.3f}")
        c2.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
        c3.metric("Balanced accuracy", f"{metrics.get('balanced_acc', 0):.1%}")
        c4.metric("Wrong-direction calls", f"{metrics.get('opposite_direction_rate', 0):.1%}")
    training = meta.get("training_data") or {}
    st.caption(f"Algorithm: {meta.get('algorithm')} · Trained on {training.get('start', '?')} to "
               f"{training.get('end', '?')} · On the same test period, simply repeating today's trend "
               "scored macro F1 0.424 and accuracy 49.2%.")


@st.cache_data(ttl=600, show_spinner=False)
def _all_tokens() -> pd.DataFrame:
    rows = []
    for t in TOKENS:
        res = get_prediction(STUDENT, t)
        rows.append({"Token": f"{t} ({TOKEN_NAMES[t]})",
                     "Trend": f"{ICONS.get(res.data['prediction'], '')} {res.data['prediction']}" if res.ok else "unavailable",
                     "Confidence": f"{res.data['confidence']:.0%}" if res.ok else "",
                     "For date": res.data.get("prediction_date", "") if res.ok else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {TOKEN_NAMES[token]} ({token})")
    st.caption("XGBoost multiclass classifier · trend 7 days ahead · Student C")

    with st.spinner("Requesting the trend from the Trend Category API (the first request can take up to a minute)..."):
        result = get_prediction(STUDENT, token)

    if not result.ok:
        show_api_error(result, "The Trend Category prediction")
        if st.button("Try again"):
            get_prediction.clear()
            st.rerun()
    else:
        data = result.data
        _current_trend(data)
        left, right = st.columns(2)
        with left:
            st.subheader("Probability of each trend")
            _probabilities(data)
        with right:
            st.subheader("Inputs used for this prediction")
            _inputs(data)
        with st.expander("How reliable is the confidence score?"):
            st.write("On the held-out test period (25 June to 14 October 2026), accuracy generally rose with "
                     "confidence, although the 60 to 70% band dipped. Only 20 predictions were above 70%.")
            st.dataframe(CONFIDENCE_BANDS.style.format({"Test accuracy": "{:.0%}"}), hide_index=True)

    st.subheader("Recent TrendScore")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
    else:
        st.altair_chart(_trend_chart(history, days=365, height=300), width="stretch")
        st.caption("TrendScore = 0.7 x (MA5 / MA20 - 1) + 0.3 x 20-day return, calculated from Kraken prices. "
                   "The model predicts which band it will be in 7 days from now.")

    st.subheader("Trend and risk features (this service's features)")
    _risk_features(token)

    st.subheader("Model information and feature importance")
    _model_details()

    with st.expander("What is the Trend Category?"):
        st.markdown(
            "The category comes from a **TrendScore** calculated 7 days ahead:\n\n"
            "`TrendScore = 0.7 x ((MA5 / MA20) - 1) + 0.3 x Return20`\n\n"
            "| TrendScore | Trend |\n|---|---|\n| below -0.10 | Strong Downtrend |\n"
            "| -0.10 to -0.02 | Downtrend |\n| -0.02 to 0.02 | Sideways |\n| 0.02 to 0.10 | Uptrend |\n"
            "| above 0.10 | Strong Uptrend |\n\n"
            "The thresholds are the same for every token, so volatile coins such as DOGE and XLM reach the "
            "strong categories far more often than BTC or TRX. In testing the model caught none of the rare "
            "Strong Downtrend weeks, so it should not be used as a crash warning on its own."
        )

    if st.toggle("Compare all 8 cryptocurrencies"):
        with st.spinner("Requesting trends for all tokens..."):
            st.dataframe(_all_tokens(), hide_index=True, width="stretch")

    st.caption(DISCLAIMER)
