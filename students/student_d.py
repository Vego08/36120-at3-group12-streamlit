"""Student D: Trading Opportunity Rating (TOR), LightGBM multiclass classification.

Owner: Student D (Vedang Gosavi, 26059271).
Required outputs (brief): Recommendation, Prediction Confidence Score, Feature Importance.
All predictions come from the Student D FastAPI service; nothing is computed by a model here.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common.api_client import get_features, get_model_metadata, get_prediction, show_api_error
from common.config import DISCLAIMER, STUDENTS, TOKEN_NAMES, TOKENS, selected_token
from common.data import load_history

STUDENT = "D"
TITLE = STUDENTS[STUDENT]["target"]

LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
COLOURS = {"Strong Sell": "#C62828", "Sell": "#EF6C00", "Hold": "#9E9E9E",
           "Buy": "#43A047", "Strong Buy": "#1B5E20"}
ICONS = {"Strong Sell": "🔴", "Sell": "🟠", "Hold": "🟡", "Buy": "🟢", "Strong Buy": "🟩"}
MEANING = {
    "Strong Sell": "Momentum, trend and volume point to a weak market condition next week.",
    "Sell": "Signals lean negative for next week.",
    "Hold": "No strong signal either way for next week.",
    "Buy": "Signals lean positive for next week.",
    "Strong Buy": ("Very strong recent momentum. Note: in testing, Strong Buy often occurred near the end of "
                   "a rally, so pull-back risk is elevated."),
}
FEATURE_TEXT = {
    "OS_proj": "Projected Opportunity Score (if prices stay flat for 7 days)",
    "OS_t": "Today's Opportunity Score",
    "MA_ratio": "Short-term vs long-term trend (MA5 / MA20)",
    "Return20": "20-day return",
    "RelativeVolume": "Today's volume vs 20-day average",
    "Volatility20": "20-day volatility",
    "MA5": "5-day moving average (price)",
    "MA20": "20-day moving average (price)",
}
FEATURE_SOURCE = {"Return20": "Student A API", "MA5": "Student B API", "MA20": "Student B API",
                  "Volatility20": "Student C API", "RelativeVolume": "TOR API (Kraken)",
                  "OS_proj": "TOR API (Kraken)", "MA_ratio": "derived (MA5 / MA20)", "OS_t": "derived"}
# Accuracy of the final model on the held-out test period (Jul to Oct 2026) by confidence band
CONFIDENCE_BANDS = pd.DataFrame({
    "Confidence": ["up to 50%", "50 to 60%", "60 to 70%", "70 to 80%", "above 80%"],
    "Test accuracy": [0.583, 0.586, 0.610, 0.647, 0.750],
})


# ---------------------------------------------------------------------------
# Home page visual
# ---------------------------------------------------------------------------
def home_visual(token: str) -> None:
    """Daily trading volume against its 20-day average (the basis of RelativeVolume)."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    df = pd.DataFrame({"Daily volume": history["volume"],
                       "20-day average": history["volume"].rolling(20).mean()}).tail(120)
    df = df.reset_index().melt("date", var_name="series", value_name="volume")
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("volume:Q", title=f"Volume ({token})"),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.condition(alt.datum.series == "20-day average", alt.value([5, 3]), alt.value([0])),
        tooltip=["date:T", "series:N", alt.Tooltip("volume:Q", format=",.0f")],
    ).properties(height=220)
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def _recommendation(data: dict) -> None:
    label = data["prediction"]
    colour = COLOURS.get(label, "#555")
    st.markdown(
        f"""<div style="border-left: 8px solid {colour}; padding: 0.8rem 1.2rem; border-radius: 6px;
        background: rgba(127,127,127,0.08); margin-bottom: 0.8rem;">
        <div style="font-size: 0.9rem; opacity: 0.8;">Rating for {data['prediction_date']}
        (based on data up to {data['as_of']})</div>
        <div style="font-size: 2.2rem; font-weight: 700; color: {colour};">{ICONS.get(label, '')} {label}</div>
        <div style="font-size: 0.95rem;">{MEANING.get(label, '')}</div></div>""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction confidence", f"{data['confidence']:.0%}")
    c2.metric("Forecast horizon", f"{data['forecast_horizon']} days")
    c3.metric("Inputs available", f"{8 - len(data.get('features_missing', []))} of 8")
    st.progress(min(max(float(data["confidence"]), 0.0), 1.0))


def _probabilities(data: dict) -> None:
    probs = data.get("probabilities", {})
    df = pd.DataFrame({"Rating": LABELS, "Probability": [probs.get(l, 0.0) for l in LABELS]})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Rating:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Rating:N", scale=alt.Scale(domain=LABELS, range=[COLOURS[l] for l in LABELS]),
                        legend=None),
        tooltip=["Rating:N", alt.Tooltip("Probability:Q", format=".1%")],
    ).properties(height=260)
    st.altair_chart(chart, width="stretch")


def _explanation(data: dict) -> None:
    contrib = data.get("feature_contributions", {})
    if not contrib:
        st.info("No explanation returned by the API.")
        return
    df = pd.DataFrame({"feature": list(contrib), "contribution": list(contrib.values())})
    df["description"] = df["feature"].map(FEATURE_TEXT).fillna(df["feature"])
    df["direction"] = df["contribution"].apply(lambda v: "supports this rating" if v >= 0 else "against it")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("contribution:Q", title=f"Contribution to '{data['prediction']}' (log-odds)"),
        y=alt.Y("description:N", sort="-x", title=None, axis=alt.Axis(labelLimit=400)),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["supports this rating", "against it"],
                                        range=["#43A047", "#C62828"])),
        tooltip=["feature:N", alt.Tooltip("contribution:Q", format="+.3f")],
    ).properties(height=34 * len(df) + 60)
    st.altair_chart(chart, width="stretch")
    top = df.iloc[df["contribution"].abs().argmax()]
    st.caption(f"The biggest driver of this prediction is **{top['description']}**. Contributions are "
               "LightGBM SHAP values for this single prediction.")


def _inputs(data: dict) -> None:
    used = data.get("features_used", {})
    rows = [{"Feature": f, "Description": FEATURE_TEXT.get(f, f), "Source": FEATURE_SOURCE.get(f, ""),
             "Value": "missing" if v is None else (f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}")}
            for f, v in used.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    missing = data.get("features_missing", [])
    if missing:
        st.info("Some inputs were unavailable and passed to the model as missing values: "
                + ", ".join(missing) + ". The model was trained to handle this, but confidence may be lower.")
    status = data.get("sibling_status", {})
    if status:
        st.caption("Feature services: " + " · ".join(f"{k.replace('_', ' ').title()}: {v}"
                                                      for k, v in status.items()))


def _volume_features(token: str) -> None:
    res = get_features(STUDENT, token)
    if res.ok:
        f = res.data.get("features", {})
        c1, c2 = st.columns(2)
        rv, sma = f.get("RelativeVolume"), f.get("VolumeSMA20")
        c1.metric("Relative volume", "n/a" if rv is None else f"{rv:.2f}x",
                  None if rv is None else ("above average" if rv >= 1 else "below average"),
                  delta_color="off")
        c2.metric(f"20-day average volume ({token})", "n/a" if sma is None else f"{sma:,.0f}")
    else:
        show_api_error(res, "Volume features")

    history = load_history(token)
    if history is not None:
        rel = (history["volume"] / history["volume"].rolling(20).mean()).rename("RelativeVolume")
        df = rel.dropna().tail(120).reset_index()
        line = alt.Chart(df).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("RelativeVolume:Q", title="Relative volume"),
            tooltip=["date:T", alt.Tooltip("RelativeVolume:Q", format=".2f")])
        rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4], color="grey").encode(y="y:Q")
        st.altair_chart((line + rule).properties(height=200), width="stretch")
        st.caption("Above 1 = more trading than usual. High volume often accompanies large moves in either "
                   "direction; in the experiments it was elevated before both Strong Sell and Strong Buy.")


def _model_details() -> None:
    res = get_model_metadata(STUDENT)
    if not res.ok:
        show_api_error(res, "Model metadata")
        return
    meta = res.data
    importance = meta.get("feature_importance", {})
    if importance:
        df = pd.DataFrame({"feature": list(importance), "importance": list(importance.values())})
        chart = alt.Chart(df).mark_bar(color="#4C72B0").encode(
            x=alt.X("importance:Q", title="Share of total split gain", axis=alt.Axis(format="%")),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature:N", alt.Tooltip("importance:Q", format=".1%")],
        ).properties(height=240)
        st.altair_chart(chart, width="stretch")
    metrics = meta.get("test_metrics") or {}
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test macro F1", f"{metrics.get('macro_f1', 0):.3f}")
        c2.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
        c3.metric("Within one class", f"{metrics.get('within_one', 0):.1%}")
        c4.metric("Severe errors", f"{metrics.get('severe_error_rate', 0):.1%}")
    st.caption(f"Algorithm: {meta.get('algorithm')} · Training period: "
               f"{' to '.join(meta.get('training_period') or [])} · Labels: "
               f"{', '.join(meta.get('classification_labels', []))}")


@st.cache_data(ttl=600, show_spinner=False)
def _all_tokens() -> pd.DataFrame:
    rows = []
    for t in TOKENS:
        res = get_prediction(STUDENT, t)
        rows.append({"Token": f"{t} ({TOKEN_NAMES[t]})",
                     "Rating": f"{ICONS.get(res.data['prediction'], '')} {res.data['prediction']}" if res.ok else "unavailable",
                     "Confidence": f"{res.data['confidence']:.0%}" if res.ok else "",
                     "For date": res.data.get("prediction_date", "") if res.ok else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {TOKEN_NAMES[token]} ({token})")
    st.caption("LightGBM multiclass classifier · rating 7 days ahead · Student D")

    with st.spinner("Requesting the rating from the TOR API (the first request can take up to a minute)..."):
        result = get_prediction(STUDENT, token)

    if not result.ok:
        show_api_error(result, "The Trading Opportunity Rating")
        if st.button("Try again"):
            get_prediction.clear()
            st.rerun()
    else:
        data = result.data
        _recommendation(data)
        left, right = st.columns(2)
        with left:
            st.subheader("Probability of each rating")
            _probabilities(data)
        with right:
            st.subheader("Why this rating?")
            _explanation(data)
        with st.expander("How reliable is the confidence score?"):
            st.write("On the held-out test period (Jul to Oct 2026), the model's accuracy increased with its "
                     "confidence:")
            st.dataframe(CONFIDENCE_BANDS.style.format({"Test accuracy": "{:.0%}"}), hide_index=True)
        with st.expander("Inputs used for this prediction"):
            _inputs(data)

    st.subheader("Trading volume (this service's features)")
    _volume_features(token)

    st.subheader("Model information and feature importance")
    _model_details()

    with st.expander("What is the Trading Opportunity Rating?"):
        st.markdown(
            "The rating is based on an **Opportunity Score** computed 7 days ahead:\n\n"
            "`OS = 0.40 x Momentum + 0.30 x Trend + 0.20 x Volume - 0.10 x Volatility penalty`\n\n"
            "| OS | Rating |\n|---|---|\n| below 0.20 | Strong Sell |\n| 0.20 to 0.40 | Sell |\n"
            "| 0.40 to 0.60 | Hold |\n| 0.60 to 0.80 | Buy |\n| 0.80 and above | Strong Buy |\n\n"
            "The score is driven mostly by 20-day momentum, so it describes market conditions rather than "
            "guaranteeing future returns."
        )

    if st.toggle("Compare all 8 cryptocurrencies"):
        with st.spinner("Requesting ratings for all tokens..."):
            st.dataframe(_all_tokens(), hide_index=True, width="stretch")

    st.caption(DISCLAIMER)
