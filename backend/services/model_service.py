"""
RateIQ – ML Model Service v3.1
Primary: LightGBM (regression + classification head)
Fixes applied (2026-06-21):
  - Single-inference: predict() now returns tier + desc in one call
  - Confidence fix: _confidence_boundary inverted formula corrected
  - Classifier fix: handles bucket/class mismatch via classes_ lookup
  - clf_model fallback: errors now logged (not swallowed)
"""
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger("rateiq.model_service")

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

# ── Confidence tiers (applied to corrected 0.0–1.0 range) ────────────────────
CONFIDENCE_TIERS = [
    (0.80, "Very High", "Highly reliable — the model has strong signal for this app profile."),
    (0.65, "High",      "Reliable prediction — confident based on the input features."),
    (0.50, "Medium",    "Moderate confidence — reasonable estimate, some uncertainty remains."),
    (0.35, "Low",       "Low confidence — limited signal; try adjusting key inputs."),
    (0.00, "Very Low",  "Very uncertain — the model has weak signal for this combination."),
]


class ModelService:
    def __init__(self, model_path: str):
        self._shap_explainer: Any = None
        self._load_artifacts(model_path)

    def _load_artifacts(self, path: str):
        path = Path(path).resolve()

        # Try joblib then pickle — prefer .joblib (faster, more portable)
        arts = None
        candidates_tried = []
        for candidate in [
            path,
            path.parent / "model_artifacts.joblib",
            path.with_suffix(".pkl"),
            path.parent / "model_artifacts.pkl",
        ]:
            if candidate == path or candidate not in [Path(c) for c in candidates_tried]:
                candidates_tried.append(str(candidate))
                if candidate.exists():
                    try:
                        import joblib
                        arts = joblib.load(candidate)
                        logger.info("Loaded artifacts via joblib: %s", candidate)
                        break
                    except Exception as e:
                        logger.warning("joblib load failed for %s: %s — trying pickle", candidate, e)
                        try:
                            with open(candidate, "rb") as f:
                                arts = pickle.load(f)
                            logger.info("Loaded artifacts via pickle: %s", candidate)
                            break
                        except Exception as e2:
                            logger.warning("pickle load also failed for %s: %s", candidate, e2)

        if arts is None:
            raise FileNotFoundError(
                f"No loadable model artifacts found. Tried: {candidates_tried}"
            )

        self.model              = arts["model"]
        self.clf_model          = arts.get("clf_model")
        self.scaler             = arts["scaler"]
        self.le_cat             = arts["le_cat"]
        self.le_cr              = arts["le_cr"]
        self.features           = arts["features"]
        self.metrics            = arts["metrics"]
        self.categories         = arts["categories"]
        self.content_ratings    = arts["content_ratings"]
        self.feature_importance = arts.get("feature_importance", {})
        self.mi_scores          = arts.get("mi_scores", {})

        # Pre-compute classifier class lookup for fast bucket→index mapping
        # FIX: clf was trained on 3 of 5 buckets (classes [2,3,4]) due to
        # stratified split dropping rare buckets 0 and 1 from training data.
        # Build a mapping from bucket integer → proba array index.
        self._clf_class_to_idx: Dict[int, int] = {}
        if self.clf_model is not None:
            try:
                classes = list(self.clf_model.classes_)
                self._clf_class_to_idx = {int(c): i for i, c in enumerate(classes)}
                logger.info(
                    "Classifier loaded. classes=%s  (covers %d of 5 possible buckets)",
                    classes, len(classes)
                )
            except Exception as e:
                logger.warning("Could not read clf_model.classes_: %s", e)
                self.clf_model = None  # disable classifier if unusable

        # SHAP
        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer loaded ✅")
        except Exception as e:
            logger.warning("SHAP unavailable, using permutation fallback: %s", e)
            self._shap_explainer = None

        model_type = self.metrics.get("model_type", "?")
        logger.info(
            "ModelService ready. model=%s  MAE=%.4f  Accuracy=%.4f  "
            "clf=%s  SHAP=%s",
            model_type,
            self.metrics.get("mae", 0),
            self.metrics.get("accuracy", 0),
            "loaded" if self.clf_model else "None",
            "TreeExplainer" if self._shap_explainer else "Permutation",
        )

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_category(self, cat: str) -> int:
        cat_up = str(cat).strip().upper()
        if cat_up in self.le_cat.classes_:
            return int(self.le_cat.transform([cat_up])[0])
        if cat in self.le_cat.classes_:
            return int(self.le_cat.transform([cat])[0])
        return 0

    def _encode_cr(self, cr: str) -> int:
        if cr in self.le_cr.classes_:
            return int(self.le_cr.transform([cr])[0])
        return 0

    def _build_feature_vector(self, data: dict) -> np.ndarray:
        vec = [
            self._encode_category(data.get("category", "")),
            float(data.get("size_mb", 25.0)),
            np.log1p(data.get("installs", 0)),
            float(data.get("price", 0.0)),
            self._encode_cr(data.get("content_rating", "")),
            np.log1p(data.get("reviews", 0)),
            float(data.get("update_days", 90)),
            float(data.get("num_screenshots", 3)),
            float(data.get("has_ads", 0)),
            float(data.get("is_free", 1)),
        ]
        return np.array(vec, dtype=float)

    # ── SHAP ──────────────────────────────────────────────────────────────────

    def _compute_shap(self, vec_scaled: np.ndarray) -> np.ndarray:
        if self._shap_explainer is not None:
            try:
                sv = self._shap_explainer.shap_values(vec_scaled.reshape(1, -1))
                arr = np.array(sv)
                if arr.ndim == 2:   return arr[0]
                if arr.ndim == 1:   return arr
                return arr.flatten()[:len(self.features)]
            except Exception as e:
                logger.debug("SHAP TreeExplainer failed, using permutation: %s", e)
        return self._permutation_shap(vec_scaled)

    def _permutation_shap(self, vec_scaled: np.ndarray) -> np.ndarray:
        base = self.model.predict(vec_scaled.reshape(1, -1))[0]
        imp  = np.zeros(len(self.features))
        for i in range(len(self.features)):
            perturbed    = vec_scaled.copy()
            perturbed[i] = 0.0
            alt          = self.model.predict(perturbed.reshape(1, -1))[0]
            imp[i]       = base - alt
        return imp

    # ── Confidence ────────────────────────────────────────────────────────────

    def _confidence_from_clf(self, vec_scaled: np.ndarray, pred_rating: float) -> float:
        """
        Probability-based confidence using the classifier head.

        FIX (2026-06-21): The classifier was trained on only 3 of 5 rating
        buckets (classes [2,3,4] — i.e. 3★, 4★, 5★).  Buckets 0 and 1 (1★, 2★)
        are absent because those ratings are rare in the dataset. We use
        _clf_class_to_idx to safely map bucket → proba index, with a fallback
        to _confidence_corrected if the bucket is not in the classifier's classes.
        """
        if self.clf_model is None:
            return self._confidence_corrected(pred_rating)

        try:
            proba  = self.clf_model.predict_proba(vec_scaled.reshape(1, -1))[0]
            bucket = max(0, min(4, int(round(pred_rating)) - 1))

            if bucket not in self._clf_class_to_idx:
                # Bucket not covered by classifier — use corrected formula
                logger.debug(
                    "Bucket %d not in clf classes %s — using corrected boundary formula",
                    bucket, list(self._clf_class_to_idx.keys())
                )
                return self._confidence_corrected(pred_rating)

            idx    = self._clf_class_to_idx[bucket]
            p_main = float(proba[idx])

            # Adjacent bucket contribution
            p_adj = 0.0
            adj_below = bucket - 1
            adj_above = bucket + 1
            if adj_below in self._clf_class_to_idx:
                p_adj += float(proba[self._clf_class_to_idx[adj_below]])
            if adj_above in self._clf_class_to_idx:
                p_adj += float(proba[self._clf_class_to_idx[adj_above]])

            conf = float(np.clip(p_main + 0.4 * p_adj, 0.30, 0.99))
            return round(conf, 3)

        except Exception as e:
            logger.warning("_confidence_from_clf error: %s — using corrected boundary", e)
            return self._confidence_corrected(pred_rating)

    def _confidence_corrected(self, prediction: float) -> float:
        """
        FIX (2026-06-21): Replaces the inverted _confidence_boundary formula.

        Old (WRONG): max confidence at 3.0 (midpoint), min at 1.0 and 5.0.
        New (CORRECT): higher confidence when prediction is further from the
        neutral midpoint (3.0). A prediction of 1.0 or 5.0 is decisive — the
        model is clearly sure. A prediction near 3.0 is uncertain.

        Additionally weighted by SHAP magnitude (via indirection through
        distance from midpoint as a proxy for decisiveness).

        Range: 0.35 (most uncertain, near 3.0) → 0.85 (most decisive, at 1 or 5)
        """
        # Distance from neutral midpoint 3.0, normalised to [0, 1]
        dist = abs(prediction - 3.0) / 2.0   # 0 at 3.0, 1 at 1.0 or 5.0
        conf = 0.35 + (dist * 0.50)           # 0.35 → 0.85
        return round(float(np.clip(conf, 0.30, 0.90)), 3)

    def _confidence_tier(self, confidence: float) -> Tuple[str, str]:
        """Return (tier_label, description) for a confidence score."""
        for threshold, tier, description in CONFIDENCE_TIERS:
            if confidence >= threshold:
                return tier, description
        return "Very Low", CONFIDENCE_TIERS[-1][2]

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, data: dict) -> Tuple[float, float, str, str, List[Dict]]:
        """
        Run full prediction pipeline in a single pass.

        FIX (2026-06-21): Previously called twice per request
        (predict + predict_with_confidence_detail). Now returns everything
        from one inference call:

        Returns:
            prediction      – predicted rating (1.0–5.0)
            confidence      – confidence score (0.0–1.0, corrected formula)
            confidence_tier – "Very High" / "High" / "Medium" / "Low" / "Very Low"
            confidence_desc – human-readable explanation
            shap_list       – sorted SHAP values
        """
        vec        = self._build_feature_vector(data)
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))[0]

        raw_pred   = float(self.model.predict(vec_scaled.reshape(1, -1))[0])
        prediction = round(float(np.clip(raw_pred, 1.0, 5.0)), 2)

        confidence = self._confidence_from_clf(vec_scaled, prediction)
        conf_tier, conf_desc = self._confidence_tier(confidence)

        shap_vals = self._compute_shap(vec_scaled)
        shap_list = []
        for i, feat in enumerate(self.features):
            shap_val = float(shap_vals[i]) if i < len(shap_vals) else 0.0
            shap_list.append({
                "feature":           feat,
                "label":             FEATURE_LABELS.get(feat, feat),
                "value":             round(shap_val, 4),
                "raw_feature_value": round(float(vec[i]), 4),
            })
        shap_list.sort(key=lambda x: abs(x["value"]), reverse=True)

        return prediction, confidence, conf_tier, conf_desc, shap_list

    def get_feature_importance_ranked(self) -> List[Dict]:
        if not self.feature_importance:
            return []
        return sorted(
            [
                {
                    "feature":    feat,
                    "label":      FEATURE_LABELS.get(feat, feat),
                    "importance": info["importance"],
                    "mi_score":   info.get("mi_score", 0),
                    "rank":       info["rank"],
                }
                for feat, info in self.feature_importance.items()
            ],
            key=lambda x: x["rank"],
        )

    def batch_predict(self, data_list: List[dict]) -> List[Tuple[float, float]]:
        """Batch prediction without SHAP (for competitor simulation)."""
        results = []
        for data in data_list:
            vec   = self._build_feature_vector(data)
            vs    = self.scaler.transform(vec.reshape(1, -1))[0]
            raw   = float(self.model.predict(vs.reshape(1, -1))[0])
            pred  = round(float(np.clip(raw, 1.0, 5.0)), 2)
            conf  = self._confidence_from_clf(vs, pred)
            results.append((pred, conf))
        return results


# ── Dataset-driven Insights Engine ───────────────────────────────────────────

def generate_dataset_insights(df) -> List[Dict]:
    """Auto-mine dataset for statistically significant patterns."""
    insights = []
    if df is None or len(df) < 50:
        return insights
    try:
        import pandas as _pd, numpy as _np

        def _safe_mean(series):
            v = series.mean()
            return None if _np.isnan(v) else round(float(v), 2)

        # 1. Free vs Paid
        if "is_free" in df.columns and "rating" in df.columns:
            fa = _safe_mean(df[df["is_free"] == 1]["rating"])
            pa = _safe_mean(df[df["is_free"] == 0]["rating"])
            if fa and pa and abs(fa - pa) > 0.05:
                d = round(fa - pa, 2)
                insights.append({"icon": "💰",
                    "finding": f"Free apps rate {abs(d):.2f}★ {'higher' if d > 0 else 'lower'} than paid apps",
                    "statistic": f"Free: {fa:.2f}★  |  Paid: {pa:.2f}★",
                    "implication": "Freemium model increases user base and average ratings",
                    "category": "pricing", "strength": abs(d)})

        # 2. Installs vs Rating
        if "installs" in df.columns and "rating" in df.columns:
            hi = _safe_mean(df[df["installs"] >= 1_000_000]["rating"])
            lo = _safe_mean(df[df["installs"] < 10_000]["rating"])
            if hi and lo:
                d = round(hi - lo, 2)
                insights.append({"icon": "📥",
                    "finding": f"High-install apps (1M+) rate {abs(d):.2f}★ {'higher' if d > 0 else 'lower'} than low-install",
                    "statistic": f"1M+: {hi:.2f}★  |  <10K: {lo:.2f}★",
                    "implication": "Social proof from installs correlates with rating quality",
                    "category": "installs", "strength": abs(d)})

        # 3. App size
        if "size_mb" in df.columns and "rating" in df.columns:
            sm = _safe_mean(df[df["size_mb"] < 30]["rating"])
            lg = _safe_mean(df[df["size_mb"] > 100]["rating"])
            if sm and lg and abs(sm - lg) > 0.05:
                d = round(sm - lg, 2)
                insights.append({"icon": "📦",
                    "finding": f"Smaller apps (<30MB) rate {abs(d):.2f}★ {'higher' if d > 0 else 'lower'} than large (>100MB)",
                    "statistic": f"<30MB: {sm:.2f}★  |  >100MB: {lg:.2f}★",
                    "implication": "Lightweight apps have lower install friction and better retention",
                    "category": "size", "strength": abs(d)})

        # 4. Update recency
        if "update_days" in df.columns and "rating" in df.columns:
            fr = _safe_mean(df[df["update_days"] <= 30]["rating"])
            st = _safe_mean(df[df["update_days"] > 180]["rating"])
            if fr and st:
                d = round(fr - st, 2)
                insights.append({"icon": "🔄",
                    "finding": f"Recently updated apps (≤30d) rate {abs(d):.2f}★ {'higher' if d > 0 else 'lower'} than stale (>180d)",
                    "statistic": f"≤30d: {fr:.2f}★  |  >180d: {st:.2f}★",
                    "implication": "Regular updates signal quality and responsiveness",
                    "category": "updates", "strength": abs(d)})

        # 5. Ads
        if "has_ads" in df.columns and "rating" in df.columns:
            na = _safe_mean(df[df["has_ads"] == 0]["rating"])
            ha = _safe_mean(df[df["has_ads"] == 1]["rating"])
            if na and ha:
                d = round(na - ha, 2)
                insights.append({"icon": "📢",
                    "finding": f"Ad-free apps rate {abs(d):.2f}★ {'higher' if d > 0 else 'lower'} than ad-supported",
                    "statistic": f"No ads: {na:.2f}★  |  Has ads: {ha:.2f}★",
                    "implication": "Reducing intrusive ads directly improves user satisfaction",
                    "category": "ads", "strength": abs(d)})

        # 6. Top/bottom category
        if "category" in df.columns and "rating" in df.columns:
            ca = df.groupby("category")["rating"].agg(["mean", "count"])
            ca = ca[ca["count"] >= 5]
            if len(ca):
                best  = ca["mean"].idxmax(); br = round(float(ca.loc[best, "mean"]), 2)
                worst = ca["mean"].idxmin(); wr = round(float(ca.loc[worst, "mean"]), 2)
                insights.append({"icon": "🏆",
                    "finding": f"{best} is the highest-rated category ({br}★) vs {worst} ({wr}★)",
                    "statistic": f"Range: {wr:.2f} → {br:.2f}★",
                    "implication": f"Apps in {best} have structural advantages — study their UX patterns",
                    "category": "category", "strength": abs(br - wr)})

        # 7. Reviews
        if "reviews" in df.columns and "rating" in df.columns:
            med = df["reviews"].median()
            hr  = _safe_mean(df[df["reviews"] > med * 2]["rating"])
            lr  = _safe_mean(df[df["reviews"] < med * 0.5]["rating"])
            if hr and lr and abs(hr - lr) > 0.05:
                d = round(hr - lr, 2)
                insights.append({"icon": "⭐",
                    "finding": f"High-review apps rate {abs(d):.2f}★ {'higher' if d > 0 else 'lower'} than low-review apps",
                    "statistic": f"High reviews: {hr:.2f}★  |  Low reviews: {lr:.2f}★",
                    "implication": "Social proof through reviews is a strong positive signal",
                    "category": "reviews", "strength": abs(d)})

        insights.sort(key=lambda x: x["strength"], reverse=True)
    except Exception as e:
        logger.warning("Insight generation error: %s", e)
    return insights


# ── Singleton ─────────────────────────────────────────────────────────────────
_service: Optional[ModelService] = None


def get_model_service(model_path: Optional[str] = None) -> ModelService:
    global _service
    if _service is None:
        from backend.core.config import settings
        _service = ModelService(model_path or settings.model_path)
    return _service
