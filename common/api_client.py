"""HTTP calls to the FastAPI services, with timeouts and graceful error handling.

No model inference happens in Streamlit: every prediction comes from an API call.
A failing or sleeping API never crashes a page; it returns an ApiResult with ok=False.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

from common.config import (
    FORECAST_HORIZON,
    HEALTH_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    STUDENTS,
    api_url,
)


@dataclass
class ApiResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None
    elapsed_s: float | None = None


def call_api(student: str, method: str, endpoint: str, payload: dict | None = None,
             timeout: float = REQUEST_TIMEOUT_SECONDS) -> ApiResult:
    """Call one endpoint of a student's API and never raise."""
    base = api_url(student)
    if not base:
        return ApiResult(ok=False, error="API URL not configured yet.")

    start = time.perf_counter()
    try:
        response = requests.request(method, f"{base}{endpoint}", json=payload, timeout=timeout)
    except requests.Timeout:
        return ApiResult(ok=False, error=f"No response within {timeout:.0f} s. The service may be "
                                         "waking up on Render; please try again shortly.")
    except requests.ConnectionError:
        return ApiResult(ok=False, error="Service unreachable (connection error).")
    except requests.RequestException as exc:
        return ApiResult(ok=False, error=f"Request failed: {exc}")
    elapsed = time.perf_counter() - start

    try:
        body = response.json()
    except ValueError:
        body = None

    if response.status_code != 200:
        detail = body.get("error") or body.get("detail") if isinstance(body, dict) else None
        return ApiResult(ok=False, data=body if isinstance(body, dict) else None,
                         status_code=response.status_code, elapsed_s=elapsed,
                         error=f"HTTP {response.status_code}: {detail or response.text[:200]}")
    if not isinstance(body, dict):
        return ApiResult(ok=False, status_code=response.status_code, elapsed_s=elapsed,
                         error="Response was not a JSON object.")
    return ApiResult(ok=True, data=body, status_code=200, elapsed_s=elapsed)


@st.cache_data(ttl=600, show_spinner=False)
def get_prediction(student: str, token: str) -> ApiResult:
    return call_api(student, "POST", "/predict", {"token": token, "forecast_horizon": FORECAST_HORIZON})


@st.cache_data(ttl=600, show_spinner=False)
def get_features(student: str, token: str) -> ApiResult:
    return call_api(student, "POST", "/features", {"token": token})


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_metadata(student: str) -> ApiResult:
    return call_api(student, "GET", "/model-metadata")


def check_health(student: str) -> ApiResult:
    return call_api(student, "GET", "/health", timeout=HEALTH_TIMEOUT_SECONDS)


def service_status_table() -> pd.DataFrame:
    """One row per API with its health status, used on the home page."""
    rows = []
    for student, info in STUDENTS.items():
        res = check_health(student)
        rows.append({
            "Student": student,
            "Service": info["target"],
            "Status": "Online" if res.ok else "Unavailable",
            "Response time (s)": round(res.elapsed_s, 2) if res.elapsed_s else None,
            "Detail": "" if res.ok else res.error,
        })
    return pd.DataFrame(rows)


def show_api_error(result: ApiResult, what: str) -> None:
    """Consistent, friendly message when an API call fails; the page keeps working."""
    st.warning(f"{what} is currently unavailable. {result.error or ''}", icon="⚠️")


def render_prediction(student: str, token: str, label: str) -> ApiResult:
    """Standard prediction card: value, confidence (classification) and prediction date."""
    with st.spinner(f"Requesting {label} from Student {student}'s API..."):
        result = get_prediction(student, token)
    if not result.ok:
        show_api_error(result, label)
        return result

    data = result.data
    cols = st.columns(3)
    value = data.get("prediction")
    cols[0].metric(label, f"{value:.1f}" if isinstance(value, (int, float)) else str(value))
    if "confidence" in data:
        cols[1].metric("Prediction confidence", f"{float(data['confidence']):.0%}")
    cols[2].metric("Prediction for", str(data.get("prediction_date", "")))
    missing = data.get("features_missing") or []
    if missing:
        st.info("Some inputs were unavailable and treated as missing: " + ", ".join(missing))
    return result


def render_feature_importance(student: str) -> ApiResult:
    """Bar chart of feature importance from /model-metadata."""
    result = get_model_metadata(student)
    if not result.ok:
        show_api_error(result, "Model metadata")
        return result
    importance = result.data.get("feature_importance") or {}
    if not importance:
        st.info("The API did not return feature importance.")
        return result
    series = pd.Series(importance, name="importance").sort_values(ascending=False)
    st.bar_chart(series, horizontal=True)
    return result
