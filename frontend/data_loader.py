"""
RateIQ – Dataset Loader
Handles loading of user-provided datasets:
  - cleaned dataset   (apps_cleaned.csv)
  - feature dataset   (apps_features.csv)
  - fallback          (apps.csv / synthetic generation)

Auto-detects available feature columns and normalises naming.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger("rateiq.data_loader")

# ── Canonical paths ────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

PATHS = {
    "cleaned":         os.path.join(_DATA_DIR, "apps_cleaned.csv"),
    "features":        os.path.join(_DATA_DIR, "apps_features.csv"),
    "default":         os.path.join(_DATA_DIR, "apps.csv"),
    # Real provided datasets (highest priority)
    "features_real":   os.path.join(_DATA_DIR, "App_playstore_features.csv"),
    "cleaned_real":    os.path.join(_DATA_DIR, "App_playstore_final_cleaned.csv"),
}

# ── Column aliases – maps common raw names → canonical names ──────────────────
COLUMN_ALIASES: Dict[str, str] = {
    # ── Real Play Store dataset columns ────────────────────────────────────
    "App":              "app_name",
    "Category":         "category",
    "Rating":           "rating",
    "Reviews":          "reviews",
    "Size":             "size_mb",
    "Installs":         "installs",
    "Type":             "type",
    "Price":            "price",
    "Content Rating":   "content_rating",
    "Genres":           "genres",
    "Android Ver":      "android_ver",
    "is_free":          "is_free",
    "Category_enc":     "category_enc_provided",
    "Genres_enc":       "genres_enc",
    "Content Rating_enc": "content_rating_enc_provided",
    "mean_polarity":    "mean_polarity",
    "mean_subjectivity":"mean_subjectivity",
    "positive_count":   "positive_count",
    "negative_count":   "negative_count",
    "neutral_count":    "neutral_count",
    "review_count":     "review_count",
    "log_reviews":      "log_reviews",
    "log_installs":     "log_installs",
    "sentiment_ratio":  "sentiment_ratio",
    "size_category":    "size_category",
    "size_category_enc":"size_category_enc",
    "rating_category":  "rating_category",
    "rating_category_enc": "rating_category_enc",
    # ── Generic/alternative column names ──────────────────────────────────
    "App Rating":       "rating",
    "average_rating":   "rating",
    "avg_rating":       "rating",
    "App Category":     "category",
    "genre":            "category",
    "Genre":            "category",
    "size":             "size_mb",
    "App Size":         "size_mb",
    "Size (MB)":        "size_mb",
    "size_in_mb":       "size_mb",
    "install_count":    "installs",
    "total_installs":   "installs",
    "Downloads":        "installs",
    "app_price":        "price",
    "ContentRating":    "content_rating",
    "age_rating":       "content_rating",
    "review_count":     "reviews",
    "num_reviews":      "reviews",
    "Total Reviews":    "reviews",
    "Days Since Update":"update_days",
    "days_since_update":"update_days",
    "last_updated_days":"update_days",
    "Screenshots":      "num_screenshots",
    "screenshot_count": "num_screenshots",
    "Has Ads":          "has_ads",
    "ad_supported":     "has_ads",
    "Free":             "is_free",
    "free_app":         "is_free",
    "category_enc":     "category_enc",
    "content_rating_enc": "content_rating_enc",
}

# Minimum required columns for EDA/display
REQUIRED_EDA_COLS = {"category", "rating"}

# Full model feature set
MODEL_FEATURE_COLS = [
    "category", "size_mb", "installs", "price", "content_rating",
    "reviews", "update_days", "num_screenshots", "has_ads", "is_free",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using alias map, then lower-strip everything else."""
    df = df.rename(columns=COLUMN_ALIASES)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def _parse_installs(val) -> Optional[int]:
    """Convert '1,000,000+' or '1M' strings to int."""
    if pd.isna(val):
        return None
    s = str(val).replace(",", "").replace("+", "").replace(" ", "").upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _parse_size(val) -> Optional[float]:
    """Convert '25.5M' / '2.1k' / '15 MB' strings to float MB."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper().replace(" ", "")
    if s.endswith("MB") or s.endswith("M"):
        return float(s.replace("MB", "").replace("M", ""))
    if s.endswith("KB") or s.endswith("K"):
        return float(s.replace("KB", "").replace("K", "")) / 1024.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_price(val) -> float:
    """Convert '$1.99' / 'Free' strings to float."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace("$", "").replace(",", "")
    if s.lower() in ("free", "0", ""):
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply type coercion and cleaning to a normalised dataframe."""
    df = _normalise_columns(df)

    # Parse flexible types
    if "installs" in df.columns:
        df["installs"] = df["installs"].apply(_parse_installs).fillna(0).astype(int)
    if "size_mb" in df.columns:
        df["size_mb"] = df["size_mb"].apply(_parse_size)
        df["size_mb"] = pd.to_numeric(df["size_mb"], errors="coerce").fillna(25.0)
    if "price" in df.columns:
        df["price"] = df["price"].apply(_parse_price)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].between(1.0, 5.0, inclusive="both")].copy()
    if "reviews" in df.columns:
        df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0).astype(int)
    if "update_days" in df.columns:
        df["update_days"] = pd.to_numeric(df["update_days"], errors="coerce").fillna(90).astype(int)
    if "num_screenshots" in df.columns:
        df["num_screenshots"] = pd.to_numeric(df["num_screenshots"], errors="coerce").fillna(3).astype(int).clip(0, 8)
    if "has_ads" in df.columns:
        df["has_ads"] = df["has_ads"].apply(lambda x: 1 if str(x).lower() in ("1","true","yes","y") else 0).astype(int)
    if "is_free" in df.columns:
        df["is_free"] = df["is_free"].apply(lambda x: 1 if str(x).lower() in ("1","true","yes","y","free") else 0).astype(int)
    elif "price" in df.columns:
        df["is_free"] = (df["price"] == 0.0).astype(int)

    # Add derived log features if not already present
    if "installs" in df.columns and "log_installs" not in df.columns:
        df["log_installs"] = np.log1p(df["installs"])
    if "reviews" in df.columns and "log_reviews" not in df.columns:
        df["log_reviews"] = np.log1p(df["reviews"])

    # Drop rows missing critical columns
    if "category" in df.columns:
        df = df[df["category"].notna() & (df["category"] != "")]
    if "rating" in df.columns:
        df = df.dropna(subset=["rating"])

    return df.reset_index(drop=True)


def _generate_synthetic(n: int = 3000) -> pd.DataFrame:
    """Minimal synthetic fallback dataset."""
    np.random.seed(42)
    cats = ["Education","Entertainment","Game","Health & Fitness",
            "Productivity","Shopping","Social","Tools","Travel & Local",
            "Finance","Photography","Music & Audio","Food & Drink"]
    df = pd.DataFrame({
        "category":       np.random.choice(cats, n),
        "size_mb":        np.random.uniform(5, 120, n).round(1),
        "installs":       np.random.choice([1_000,10_000,100_000,1_000_000,10_000_000], n),
        "price":          np.random.choice([0.0]*8 + [0.99,1.99,4.99], n),
        "content_rating": np.random.choice(["Everyone","Teen","Mature 17+"], n),
        "reviews":        np.random.randint(10, 200_000, n),
        "update_days":    np.random.randint(1, 500, n),
        "num_screenshots":np.random.randint(1, 8, n),
        "has_ads":        np.random.randint(0, 2, n),
        "is_free":        np.ones(n, dtype=int),
        "rating":         np.round(np.clip(np.random.normal(4.0, 0.5, n), 1, 5), 1),
    })
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def detect_available_datasets() -> Dict[str, bool]:
    """Return which dataset files are available on disk."""
    return {k: os.path.exists(v) for k, v in PATHS.items()}


def detect_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Auto-detect which feature columns are present in a dataframe.
    Returns categorised column lists.
    """
    all_cols = set(df.columns)
    return {
        "model_ready":   [c for c in MODEL_FEATURE_COLS if c in all_cols],
        "engineered":    [c for c in ["log_installs","log_reviews","category_enc","content_rating_enc"] if c in all_cols],
        "target":        ["rating"] if "rating" in all_cols else [],
        "extra":         [c for c in all_cols if c not in MODEL_FEATURE_COLS
                          and c not in ["log_installs","log_reviews","category_enc","content_rating_enc","rating"]],
        "missing_model": [c for c in MODEL_FEATURE_COLS if c not in all_cols],
    }


def load_dataset(prefer: str = "auto", n_synthetic: int = 3000) -> Tuple[pd.DataFrame, str]:
    """
    Load the best available dataset.

    Args:
        prefer:       "cleaned" | "features" | "default" | "auto"
                      "auto" tries: cleaned → features → default → synthetic
        n_synthetic:  rows to generate if all files missing

    Returns:
        (dataframe, source_label)
    """
    order = (
        [prefer] if prefer != "auto"
        else ["features_real", "cleaned_real", "cleaned", "features", "default"]
    )

    for key in order:
        path = PATHS.get(key)
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)
                df = _clean_dataframe(df)
                if REQUIRED_EDA_COLS.issubset(df.columns) and len(df) > 0:
                    logger.info("Loaded '%s' dataset: %d rows from %s", key, len(df), path)
                    return df, key
                else:
                    logger.warning("Dataset '%s' missing required columns or empty, skipping.", key)
            except Exception as e:
                logger.warning("Failed to load '%s' (%s): %s", key, path, e)

    # Fallback: synthetic
    logger.info("No user dataset found — generating %d synthetic rows.", n_synthetic)
    return _generate_synthetic(n_synthetic), "synthetic"


def load_cleaned_dataset() -> Tuple[pd.DataFrame, str]:
    """Load cleaned dataset preferring apps_cleaned.csv."""
    return load_dataset(prefer="cleaned")


def load_feature_dataset() -> Tuple[pd.DataFrame, str]:
    """Load feature-engineered dataset preferring apps_features.csv."""
    return load_dataset(prefer="features")


def get_dataset_summary(df: pd.DataFrame, source: str) -> Dict:
    """Return a summary dict for display in the UI."""
    col_info = detect_feature_columns(df)
    return {
        "source":          source,
        "rows":            len(df),
        "columns":         list(df.columns),
        "model_ready_cols": col_info["model_ready"],
        "engineered_cols": col_info["engineered"],
        "missing_cols":    col_info["missing_model"],
        "has_target":      bool(col_info["target"]),
        "avg_rating":      round(df["rating"].mean(), 3) if "rating" in df.columns else None,
        "categories":      sorted(df["category"].unique().tolist()) if "category" in df.columns else [],
        "num_categories":  df["category"].nunique() if "category" in df.columns else 0,
    }
