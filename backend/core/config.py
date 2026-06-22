"""
RateIQ – Backend Configuration
Reads from environment variables with sensible defaults.

Model artifact priority (verified 2026-06-21):
  PRIMARY  → backend/models/model_artifacts.joblib  (8.2 MB, LightGBM)
  FALLBACK → backend/models/model_artifacts.pkl     (8.2 MB, same content)
  STALE    → model.joblib at project root           (72 KB, old RandomForest — NEVER load)
"""
from pydantic_settings import BaseSettings
import os

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

class Settings(BaseSettings):
    app_name: str = "RateIQ API"
    version: str  = "3.0.0"
    debug: bool   = False

    # Point explicitly to .joblib — loaded first by ModelService._load_artifacts()
    model_path: str = os.environ.get(
        "MODEL_PATH",
        os.path.abspath(os.path.join(_MODELS_DIR, "model_artifacts.joblib")),
    )

    db_url: str = os.environ.get("DB_URL", "sqlite:///./rateiq.db")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
