"""
RateIQ – API Client v2.1
Handles all communication between Streamlit frontend and FastAPI backend.
Fix (2026-06-21): api_get now accepts params dict — no more embedded query strings.
"""
import logging
import os
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger("rateiq.client")

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000") + "/api/v1"
TIMEOUT  = 12


class APIError(Exception):
    pass


def _post(endpoint: str, payload: dict, timeout: int = TIMEOUT) -> dict:
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot connect to backend. Is FastAPI running on port 8000?")
    except requests.exceptions.Timeout:
        raise APIError("Request timed out.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:   detail = e.response.json().get("detail", "")
        except Exception: pass
        raise APIError(f"API error {e.response.status_code}: {detail or str(e)}")
    except Exception as e:
        raise APIError(str(e))


def _get(endpoint: str, params: Optional[dict] = None, timeout: int = TIMEOUT) -> Any:
    """
    GET request.  params dict is passed as query parameters — do NOT embed
    ?key=val in the endpoint string.
    """
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot connect to backend. Is FastAPI running on port 8000?")
    except requests.exceptions.Timeout:
        raise APIError("Request timed out.")
    except Exception as e:
        raise APIError(str(e))


# ── Public helpers ────────────────────────────────────────────────────────────

def predict(payload: dict) -> dict:
    return _post("/predict", payload)


def chat(query: str, app_data: Optional[dict] = None,
         prediction_data: Optional[dict] = None,
         chat_history: Optional[List[Dict]] = None) -> dict:
    return _post("/chat", {
        "query": query,
        "app_data": app_data,
        "prediction_data": prediction_data,
        "chat_history": chat_history or [],
    }, timeout=20)


def competitor_analysis(app_data: dict, predicted_rating: Optional[float] = None) -> dict:
    payload = {"app_data": app_data}
    if predicted_rating is not None:
        payload["predicted_rating"] = predicted_rating
    return _post("/competitor-analysis", payload)


def trend_boost(category: str, base_prediction: float) -> dict:
    return _post("/trend", {"category": category, "base_prediction": base_prediction})


def get_meta() -> dict:
    return _get("/meta")


def get_history(limit: int = 50) -> list:
    # FIX: use params dict, not embedded query string
    return _get("/history", params={"limit": limit})


def get_feature_importance() -> dict:
    return _get("/feature-importance")


def get_dataset_insights() -> dict:
    return _get("/dataset-insights")


def health_check() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
