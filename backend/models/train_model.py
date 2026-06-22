"""
RateIQ – Model Training Pipeline v3
Primary model: LightGBM (regression + classification for confidence)
Fallback:      XGBoost → GradientBoosting
Pipeline:      preprocessing → feature selection → training → SHAP → evaluation
Dataset:       Auto-detects provided CSVs before generating synthetic data
Artifacts:     Saved via joblib (model, scaler, encoders, metrics, feature importance)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import pickle

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix,
    classification_report,
)
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.pipeline import Pipeline

# ── Column name mappings for the real Play Store datasets ─────────────────────
REAL_COLUMN_MAP = {
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
    "Category_enc":     "category_enc_raw",
    "Genres_enc":       "genres_enc",
    "Content Rating_enc": "content_rating_enc_raw",
    "mean_polarity":    "mean_polarity",
    "mean_subjectivity":"mean_subjectivity",
    "positive_count":   "positive_count",
    "negative_count":   "negative_count",
    "neutral_count":    "neutral_count",
    "review_count":     "review_count",
    "log_reviews":      "log_reviews_raw",
    "log_installs":     "log_installs_raw",
    "sentiment_ratio":  "sentiment_ratio",
    "size_category":    "size_category",
    "size_category_enc":"size_category_enc",
    "rating_category":  "rating_category",
    "rating_category_enc": "rating_category_enc",
}

FEATURES = [
    "category_enc", "size_mb", "log_installs", "price",
    "content_rating_enc", "log_reviews", "update_days",
    "num_screenshots", "has_ads", "is_free",
]

FEATURE_LABELS = {
    "category_enc":       "App Category",
    "size_mb":            "App Size (MB)",
    "log_installs":       "Install Count",
    "price":              "Price (USD)",
    "content_rating_enc": "Content Rating",
    "log_reviews":        "Review Count",
    "update_days":        "Days Since Update",
    "num_screenshots":    "Screenshots Count",
    "has_ads":            "Contains Ads",
    "is_free":            "Is Free",
}


# ── Synthetic fallback data ───────────────────────────────────────────────────

def generate_synthetic_data(n: int = 8000) -> pd.DataFrame:
    categories = [
        "Art & Design", "Auto & Vehicles", "Beauty", "Books & Reference",
        "Business", "Comics", "Communication", "Dating", "Education",
        "Entertainment", "Events", "Finance", "Food & Drink", "Health & Fitness",
        "House & Home", "Libraries & Demo", "Lifestyle", "Maps & Navigation",
        "Medical", "Music & Audio", "News & Magazines", "Parenting",
        "Personalization", "Photography", "Productivity", "Shopping",
        "Social", "Sports", "Tools", "Travel & Local", "Video Players",
        "Weather", "Game",
    ]
    content_ratings = ["Everyone", "Everyone 10+", "Teen", "Mature 17+", "Adults only 18+"]

    cat = np.random.choice(categories, n)
    size_mb = np.random.uniform(1, 200, n)
    installs = np.random.choice(
        [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000,
         500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000],
        n,
    )
    price    = np.random.choice([0.0] * 8 + [0.99, 1.99, 2.99, 4.99, 9.99], n)
    cr       = np.random.choice(content_ratings, n)
    reviews  = np.clip(np.round(np.random.exponential(12_000, n)).astype(int), 1, 5_000_000)
    upd_days = np.random.randint(1, 730, n)
    num_ss   = np.random.randint(1, 9, n)
    has_ads  = np.random.randint(0, 2, n)
    is_free  = (price == 0.0).astype(int)

    cat_bias = np.where(np.isin(cat, ["Education", "Health & Fitness", "Medical"]), 0.18,
               np.where(np.isin(cat, ["Game", "Social", "Dating"]), -0.18, 0.0))
    rating = np.clip(
        3.5 + cat_bias
        + np.log1p(reviews) * 0.06
        + np.log1p(installs) * 0.025
        - has_ads * 0.15
        + is_free * 0.10
        - (upd_days / 730) * 0.30
        + num_ss * 0.025
        - (size_mb / 200) * 0.10
        + np.random.normal(0, 0.26, n),
        1.0, 5.0,
    ).round(1)

    return pd.DataFrame({
        "category": cat, "size_mb": size_mb.round(2), "installs": installs,
        "price": price, "content_rating": cr, "reviews": reviews,
        "update_days": upd_days, "num_screenshots": num_ss,
        "has_ads": has_ads, "is_free": is_free, "rating": rating,
    })


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_installs(val):
    if pd.isna(val): return 0
    s = str(val).replace(",", "").replace("+", "").replace(" ", "").upper()
    try:
        if s.endswith("M"): return int(float(s[:-1]) * 1_000_000)
        if s.endswith("K"): return int(float(s[:-1]) * 1_000)
        return int(float(s))
    except Exception:
        return 0


def _parse_size(val):
    if pd.isna(val): return 25.0
    s = str(val).strip().upper().replace(" ", "")
    try:
        if s.endswith("MB") or s.endswith("M"):
            return float(s.replace("MB", "").replace("M", ""))
        if s.endswith("KB") or s.endswith("K"):
            return float(s.replace("KB", "").replace("K", "")) / 1024.0
        return float(s)
    except Exception:
        return 25.0


def _parse_price(val):
    if pd.isna(val): return 0.0
    s = str(val).strip().replace("$", "").replace(",", "")
    if s.lower() in ("free", "0", ""): return 0.0
    try: return float(s)
    except Exception: return 0.0


def load_real_dataset(data_dir: str):
    """
    Try to load the provided Play Store datasets.
    Priority: App_playstore_features.csv → App_playstore_final_cleaned.csv
              → apps_cleaned.csv → apps_features.csv → apps.csv
    Returns (df, source_label) or (None, None).
    """
    candidates = {
        "features_real": os.path.join(data_dir, "App_playstore_features.csv"),
        "cleaned_real":  os.path.join(data_dir, "App_playstore_final_cleaned.csv"),
        "cleaned":       os.path.join(data_dir, "apps_cleaned.csv"),
        "features":      os.path.join(data_dir, "apps_features.csv"),
        "default":       os.path.join(data_dir, "apps.csv"),
    }
    for label, path in candidates.items():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            # Rename real dataset columns to canonical names
            df = df.rename(columns=REAL_COLUMN_MAP)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if "rating" in df.columns and "category" in df.columns:
                print(f"      Loaded '{label}': {len(df)} rows from {os.path.basename(path)}")
                return df, label
        except Exception as e:
            print(f"      WARNING: Could not read '{label}': {e}")
    return None, None


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, le_cat=None, le_cr=None, fit=True):
    """
    Full feature engineering pipeline.
    If fit=True: fits encoders. If fit=False: uses pre-fitted encoders.
    Returns (df_engineered, le_cat, le_cr)
    """
    df = df.copy()

    # Parse raw values from real datasets
    if "size_mb" in df.columns:
        df["size_mb"] = df["size_mb"].apply(_parse_size).astype(float)
    else:
        df["size_mb"] = 25.0

    if "installs" in df.columns:
        df["installs"] = df["installs"].apply(_parse_installs).astype(int)
    else:
        df["installs"] = 100_000

    if "price" in df.columns:
        df["price"] = df["price"].apply(_parse_price).astype(float)
    else:
        df["price"] = 0.0

    if "reviews" in df.columns:
        df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0).astype(int)
    else:
        df["reviews"] = 0

    # Derived fields
    if "is_free" not in df.columns:
        df["is_free"] = (df["price"] == 0.0).astype(int)
    else:
        df["is_free"] = pd.to_numeric(df["is_free"], errors="coerce").fillna(1).astype(int)

    if "has_ads" not in df.columns:
        df["has_ads"] = df["type"].apply(lambda x: 0 if str(x).strip().lower() == "free" else 0) \
            if "type" in df.columns else 0
    else:
        df["has_ads"] = pd.to_numeric(df["has_ads"], errors="coerce").fillna(0).astype(int)

    if "update_days" not in df.columns:
        df["update_days"] = 90  # default

    if "num_screenshots" not in df.columns:
        df["num_screenshots"] = 3  # default

    if "content_rating" not in df.columns:
        df["content_rating"] = "Everyone"

    # Clip & clean
    df["size_mb"]         = df["size_mb"].clip(0.1, 2000)
    df["update_days"]     = pd.to_numeric(df.get("update_days", 90), errors="coerce").fillna(90).astype(int).clip(0, 3650)
    df["num_screenshots"] = pd.to_numeric(df.get("num_screenshots", 3), errors="coerce").fillna(3).astype(int).clip(0, 8)
    df["category"]        = df["category"].astype(str).str.strip().str.upper()
    df["content_rating"]  = df["content_rating"].astype(str).str.strip()

    # Log transforms
    df["log_installs"] = np.log1p(df["installs"])
    df["log_reviews"]  = np.log1p(df["reviews"])

    # Label encoding
    if fit:
        le_cat = LabelEncoder()
        le_cr  = LabelEncoder()
        df["category_enc"]       = le_cat.fit_transform(df["category"])
        df["content_rating_enc"] = le_cr.fit_transform(df["content_rating"])
    else:
        df["category_enc"] = df["category"].apply(
            lambda c: int(le_cat.transform([c])[0]) if c in le_cat.classes_ else 0
        )
        df["content_rating_enc"] = df["content_rating"].apply(
            lambda c: int(le_cr.transform([c])[0]) if c in le_cr.classes_ else 0
        )

    return df, le_cat, le_cr


# ── Rating bucketing for classification head ──────────────────────────────────

def rating_to_bucket(r: float) -> int:
    """
    Convert continuous rating to 5 ordinal buckets for classification.
    Used to compute probability-based confidence scores.
    1→0, 2→1, 3→2, 4→3, 5→4
    """
    return max(0, min(4, int(round(r)) - 1))


# ── Main training function ────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  RateIQ – Model Training Pipeline v3 (LightGBM)")
    print("=" * 60)

    # Support running from project root OR from backend/models/
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # Always resolve data dir relative to project root (backend/models → backend → project)
    _project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
    data_dir = os.path.join(_project_root, "data")
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(_script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── [1/6] Load data ────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    df_raw, dataset_source = load_real_dataset(data_dir)

    if df_raw is None:
        print("      No provided dataset found — generating synthetic data (8,000 rows)...")
        df_raw = generate_synthetic_data(8_000)
        dataset_source = "synthetic"

    # Save a working copy for EDA
    apps_path = os.path.join(data_dir, "apps.csv")
    df_raw.to_csv(apps_path, index=False)
    print(f"      Working copy → data/apps.csv")

    # ── [2/6] Feature engineering ──────────────────────────────────────────
    print("\n[2/6] Feature engineering...")
    df_raw["rating"] = pd.to_numeric(df_raw.get("rating", 4.0), errors="coerce")
    df_raw = df_raw[df_raw["rating"].between(1.0, 5.0)].dropna(subset=["category", "rating"])
    df_raw = df_raw.reset_index(drop=True)

    df, le_cat, le_cr = engineer_features(df_raw, fit=True)
    print(f"      Clean rows: {len(df)} | Categories: {len(le_cat.classes_)}")
    print(f"      Features: {FEATURES}")
    print(f"      Rating distribution:\n{df['rating'].value_counts().sort_index().to_dict()}")

    X = df[FEATURES].values
    y = df["rating"].values
    y_bucket = np.array([rating_to_bucket(r) for r in y])

    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X, y, y_bucket, test_size=0.20, random_state=42, stratify=y_bucket
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── [3/6] Feature selection (mutual information) ───────────────────────
    print("\n[3/6] Feature selection...")
    mi_scores = mutual_info_regression(X_train_s, y_train, random_state=42)
    mi_ranking = sorted(zip(FEATURES, mi_scores), key=lambda x: x[1], reverse=True)
    print("      Mutual Information ranking:")
    for feat, score in mi_ranking:
        bar = "█" * int(score * 40)
        print(f"        {FEATURE_LABELS.get(feat, feat):25s}  {score:.4f}  {bar}")

    # ── [4/6] LightGBM training ────────────────────────────────────────────
    print("\n[4/6] Training LightGBM regressor...")
    model_type = "unknown"
    model      = None

    try:
        import lightgbm as lgb

        # Compute sample weights to handle rating imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", yb_train)

        lgb_params = {
            "objective":        "regression",
            "metric":           ["mae", "rmse"],
            "n_estimators":     800,
            "max_depth":        7,
            "num_leaves":       63,
            "learning_rate":    0.03,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq":     5,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "min_child_samples":20,
            "random_state":     42,
            "n_jobs":           -1,
            "verbose":          -1,
        }
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train_s, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test_s, y_test)],
            callbacks=[lgb.log_evaluation(period=100)],
        )
        model_type = "LightGBM"
        print("      LightGBM training complete ✅")

    except ImportError:
        print("      LightGBM not installed, trying XGBoost...")
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                subsample=0.80, colsample_bytree=0.80,
                reg_alpha=0.1, reg_lambda=1.5, random_state=42,
                n_jobs=-1, tree_method="hist", verbosity=0,
            )
            model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
            model_type = "XGBoost"
            print("      XGBoost training complete ✅")
        except ImportError:
            print("      Falling back to GradientBoostingRegressor...")
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.04,
                subsample=0.80, random_state=42,
            )
            model.fit(X_train_s, y_train)
            model_type = "GradientBoosting"
            print("      GradientBoosting training complete ✅")

    # ── [4b/6] LightGBM classifier for probability-based confidence ────────
    clf_model  = None
    clf_type   = None
    try:
        import lightgbm as lgb
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(yb_train)
        cw = compute_class_weight("balanced", classes=classes, y=yb_train)
        cw_dict = dict(zip(classes.tolist(), cw.tolist()))

        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=5, metric="multi_logloss",
            n_estimators=400, max_depth=6, num_leaves=31, learning_rate=0.05,
            feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=5,
            class_weight=cw_dict, random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_train_s, yb_train, eval_set=[(X_test_s, yb_test)],
                callbacks=[lgb.log_evaluation(period=200)])
        clf_model = clf
        clf_type  = "LightGBM-Classifier"
        yb_pred = clf.predict(X_test_s)
        print(f"\n      Classifier accuracy: {accuracy_score(yb_test, yb_pred):.4f}")
        print(f"      Classifier F1 (macro): {f1_score(yb_test, yb_pred, average='macro', zero_division=0):.4f}")
    except Exception as e:
        print(f"      Classifier training skipped: {e}")

    # ── [5/6] Evaluation ───────────────────────────────────────────────────
    print("\n[5/6] Evaluating regression model...")
    y_pred = model.predict(X_test_s)
    y_pred = np.clip(y_pred, 1.0, 5.0)

    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    # Accuracy as "within ±0.5 stars"
    within_half = float(np.mean(np.abs(y_test - y_pred) <= 0.5))
    within_one  = float(np.mean(np.abs(y_test - y_pred) <= 1.0))

    # Classification metrics from regression buckets
    yb_pred_from_reg = np.array([rating_to_bucket(float(p)) for p in y_pred])
    acc = float(accuracy_score(yb_test, yb_pred_from_reg))
    f1  = float(f1_score(yb_test, yb_pred_from_reg, average="weighted", zero_division=0))
    cm  = confusion_matrix(yb_test, yb_pred_from_reg).tolist()

    print(f"\n      ┌─ Regression ─────────────────────────────────────┐")
    print(f"      │  MAE   : {mae:.4f}                               │")
    print(f"      │  RMSE  : {rmse:.4f}                               │")
    print(f"      │  R²    : {r2:.4f}                               │")
    print(f"      │  ±0.5★ : {within_half*100:.1f}%                               │")
    print(f"      │  ±1.0★ : {within_one*100:.1f}%                               │")
    print(f"      ├─ Classification ─────────────────────────────────┤")
    print(f"      │  Accuracy  : {acc:.4f}                          │")
    print(f"      │  F1 (wtd)  : {f1:.4f}                          │")
    print(f"      └──────────────────────────────────────────────────┘")

    # ── [5b/6] Feature importance ──────────────────────────────────────────
    feature_importance = {}
    try:
        if model_type == "LightGBM":
            raw_imp = model.feature_importances_
        elif model_type == "XGBoost":
            raw_imp = model.feature_importances_
        else:
            raw_imp = model.feature_importances_
        total = raw_imp.sum() if raw_imp.sum() > 0 else 1
        for feat, imp in zip(FEATURES, raw_imp):
            feature_importance[feat] = {
                "label":       FEATURE_LABELS.get(feat, feat),
                "importance":  round(float(imp / total), 5),
                "mi_score":    round(float(dict(mi_ranking).get(feat, 0)), 5),
                "rank":        0,
            }
        ranked = sorted(feature_importance.items(), key=lambda x: x[1]["importance"], reverse=True)
        for rank, (feat, _) in enumerate(ranked, 1):
            feature_importance[feat]["rank"] = rank
        print("\n      Feature importance (normalised):")
        for feat, info in ranked:
            bar = "█" * int(info["importance"] * 50)
            print(f"        {info['label']:25s}  {info['importance']:.4f}  {bar}")
    except Exception as e:
        print(f"      Feature importance unavailable: {e}")

    # ── [5c/6] SHAP ────────────────────────────────────────────────────────
    print("\n      Building SHAP explainer...")
    shap_available = False
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        _sv = explainer.shap_values(X_test_s[:10])
        print(f"      SHAP TreeExplainer ready. Sample shape: {np.array(_sv).shape} ✅")
        shap_available = True
    except Exception as e:
        print(f"      SHAP unavailable (permutation fallback will be used): {e}")

    # ── [6/6] Save artifacts ───────────────────────────────────────────────
    print("\n[6/6] Saving model artifacts...")
    artifacts = {
        "model":              model,
        "clf_model":          clf_model,
        "scaler":             scaler,
        "le_cat":             le_cat,
        "le_cr":              le_cr,
        "features":           FEATURES,
        "feature_labels":     FEATURE_LABELS,
        "feature_importance": feature_importance,
        "mi_scores":          dict(mi_ranking),
        "metrics": {
            "mae":          mae,
            "rmse":         rmse,
            "r2":           r2,
            "accuracy":     acc,
            "f1_weighted":  f1,
            "within_half_star": within_half,
            "within_one_star":  within_one,
            "confusion_matrix": cm,
            "model_type":   model_type,
            "clf_type":     clf_type,
            "dataset_source": dataset_source,
            "train_rows":   len(X_train),
            "test_rows":    len(X_test),
        },
        "categories":      list(le_cat.classes_),
        "content_ratings": list(le_cr.classes_),
        "shap_available":  shap_available,
    }

    out_pkl  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.pkl")
    out_jbl  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.joblib")

    # Save both formats
    with open(out_pkl, "wb") as f:
        pickle.dump(artifacts, f)
    joblib.dump(artifacts, out_jbl)

    print(f"      Saved (pickle)  → {out_pkl}")
    print(f"      Saved (joblib)  → {out_jbl}")
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete! [{model_type}]")
    print(f"     MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    print(f"     Accuracy={acc:.4f}  F1={f1:.4f}")
    print(f"     ±0.5★ hit-rate: {within_half*100:.1f}%")
    print(f"{'='*60}")
    return artifacts


if __name__ == "__main__":
    train()

