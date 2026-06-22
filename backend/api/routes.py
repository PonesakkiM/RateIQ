"""
RateIQ – API Router
Endpoints:
  POST /predict              – ML rating prediction + SHAP + trend
  POST /chat                 – Natural language AI advisor
  POST /competitor-analysis  – Competitor gap analyzer
  POST /trend                – Trend-aware rating boost (standalone)
  GET  /history              – Prediction history
  GET  /meta                 – Categories, content ratings, model metrics
  GET  /health               – Health check
"""
import json
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.schemas import (
    PredictRequest, PredictResponse, ShapValue, TrendAdjustment,
    ChatRequest, ChatResponse, IssueItem, RecommendationCard,
    CompetitorRequest, CompetitorResponse, CompetitorApp,
    FeatureGap, PerformanceGap, CompetitorSummary,
    TrendRequest, HistoryItem, MetaResponse,
    FeatureImportanceItem, FeatureImportanceResponse,
    DatasetInsightItem, DatasetInsightsResponse,
)
from backend.db.database import get_db, PredictionLog, ChatLog, CompetitorAnalysisLog
from backend.services.model_service import get_model_service, generate_dataset_insights
from backend.services.competitor_service import analyze_competitors
from backend.services.advisor_service import process_chat
from backend.services.trend_service import compute_trend_adjustment, get_all_category_trends

logger = logging.getLogger("rateiq.api")
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_recommendation(shap_list: list, prediction: float, request: PredictRequest) -> str:
    """Generate top-2 actionable developer recommendations from SHAP values."""
    tips = []
    negatives = [s for s in shap_list if s["value"] < -0.02]
    if negatives:
        worst = negatives[0]
        feat = worst["feature"]
        if "ads" in feat:
            tips.append("Reducing or removing ads can recover up to 0.18 ★ — consider a rewarded-ads or ad-free paid tier.")
        elif "update" in feat:
            tips.append("Ship an update within 2 weeks. Regular updates signal quality and boost algorithm ranking.")
        elif "screenshots" in feat:
            tips.append("Add 5-8 high-quality store screenshots with feature captions to improve listing conversion.")
        elif "price" in feat:
            tips.append("Consider a freemium model — 80%+ of top-rated apps offer a free tier.")
        elif "reviews" in feat:
            tips.append("Use in-app review prompts via Google's ReviewManager API after positive user moments.")
        elif "installs" in feat:
            tips.append("Invest in ASO (keywords, icon, screenshots) to improve organic discovery and install rate.")

    if request.has_ads and prediction < 4.0:
        tips.append("Apps with ads average 0.15 ★ lower — explore an ad-free paid tier or rewarded ads.")
    if request.update_days > 180:
        tips.append(f"Your app hasn't been updated in {request.update_days} days. Stale apps lose user trust and algorithm rank.")
    if request.num_screenshots < 3:
        tips.append("Low screenshot count hurts first impressions. Add at least 5 store screenshots.")
    if request.reviews < 100:
        tips.append("Prompt users to review after completing a key action. More social proof lifts ratings significantly.")

    if not tips:
        if prediction >= 4.3:
            tips.append("Great position! Maintain update cadence and keep engaging with user reviews.")
        else:
            tips.append("Focus on user feedback, regular updates, and response to reviews to push above 4.0.")

    return " | ".join(tips[:2])


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict app rating.
    Returns: predicted rating, confidence, SHAP values, trend adjustment, recommendation.
    """
    try:
        svc = get_model_service()
        prediction, confidence, conf_tier, conf_desc, shap_list = svc.predict(request.to_dict())

        # Trend adjustment
        trend_data = compute_trend_adjustment(request.category, prediction)
        trend_obj = TrendAdjustment(
            base_prediction=trend_data["base_prediction"],
            trend_adjustment=trend_data["trend_adjustment"],
            adjusted_rating=trend_data["adjusted_rating"],
            market_stage=trend_data["market_stage"],
            competition_level=trend_data["competition_level"],
            yoy_growth=trend_data["yoy_growth"],
            popularity_index=trend_data["popularity_index"],
            explanation=trend_data["explanation"],
            stage_advice=trend_data["stage_advice"],
            adjustment_breakdown=trend_data["adjustment_breakdown"],
        )

        # FIX: confidence tier already returned from single predict() call
        # No second svc.predict_with_confidence_detail() call needed.

        recommendation = _generate_recommendation(shap_list, prediction, request)

        # Persist to DB
        log = PredictionLog(
            input_features=json.dumps(request.to_dict()),
            prediction=prediction,
            confidence=confidence,
            trend_adjusted=trend_data["adjusted_rating"],
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            confidence_tier=conf_tier,
            confidence_desc=conf_desc,
            shap_values=[ShapValue(**s) for s in shap_list],
            model_metrics=svc.metrics,
            recommendation=recommendation,
            trend=trend_obj,
        )
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# POST /chat
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse, tags=["Advisor"])
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Natural language AI advisor.
    Acts as an AI Product Manager — analyzes query against app data and ML results.
    """
    try:
        app_dict = request.app_data.to_dict() if request.app_data else {}
        history_dicts = [m.model_dump() for m in request.chat_history] if request.chat_history else []

        result = process_chat(
            query=request.query,
            app_data=app_dict,
            prediction_data=request.prediction_data,
            chat_history=history_dicts,
        )

        # Persist chat log
        log = ChatLog(
            query=request.query,
            response=json.dumps(result),
            detected_intents=json.dumps(result.get("detected_intents", [])),
            app_context=json.dumps(app_dict) if app_dict else None,
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        return ChatResponse(
            response=result["response"],
            detected_intents=result["detected_intents"],
            issues_found=result["issues_found"],
            issue_summary=[IssueItem(**i) for i in result["issue_summary"]],
            recommendations=[RecommendationCard(**r) for r in result["recommendations"]],
            follow_up_questions=result["follow_up_questions"],
            confidence=result["confidence"],
        )
    except Exception as e:
        logger.exception("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# POST /competitor-analysis
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/competitor-analysis", response_model=CompetitorResponse, tags=["Competitor"])
async def competitor_analysis(request: CompetitorRequest, db: Session = Depends(get_db)):
    """
    Competitor Gap Analyzer.
    Finds similar apps, computes feature gaps, performance gaps, and actionable insights.
    """
    try:
        app_dict = request.app_data.to_dict()

        # Get predicted rating if not provided — run quick prediction
        if request.predicted_rating:
            app_dict["predicted_rating"] = request.predicted_rating
        else:
            try:
                svc = get_model_service()
                pred, _, _ = svc.predict(app_dict)
                app_dict["predicted_rating"] = pred
            except Exception:
                app_dict["predicted_rating"] = 3.8  # fallback

        result = analyze_competitors(app_dict)

        # Persist log
        log = CompetitorAnalysisLog(
            app_data=json.dumps(app_dict),
            analysis_result=json.dumps({k: v for k, v in result.items() if k != "similar_apps"}),
            category=app_dict.get("category"),
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

        return CompetitorResponse(
            category=result["category"],
            similar_apps=[CompetitorApp(**a) for a in result["similar_apps"]],
            top_competitors=[CompetitorApp(**a) for a in result["top_competitors"]],
            feature_gaps=[FeatureGap(**g) for g in result["feature_gaps"]],
            performance_gaps=[PerformanceGap(**g) for g in result["performance_gaps"]],
            insights=result["insights"],
            summary=CompetitorSummary(**result["summary"]),
        )
    except Exception as e:
        logger.exception("Competitor analysis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# POST /trend
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/trend", response_model=TrendAdjustment, tags=["Trend"])
async def trend_boost(request: TrendRequest):
    """
    Trend-aware rating adjustment for a given category and base prediction.
    """
    try:
        result = compute_trend_adjustment(request.category, request.base_prediction)
        return TrendAdjustment(
            base_prediction=result["base_prediction"],
            trend_adjustment=result["trend_adjustment"],
            adjusted_rating=result["adjusted_rating"],
            market_stage=result["market_stage"],
            competition_level=result["competition_level"],
            yoy_growth=result["yoy_growth"],
            popularity_index=result["popularity_index"],
            explanation=result["explanation"],
            stage_advice=result["stage_advice"],
            adjustment_breakdown=result["adjustment_breakdown"],
        )
    except Exception as e:
        logger.exception("Trend boost error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GET /history
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/history", response_model=list[HistoryItem], tags=["Data"])
async def history(limit: int = 20, db: Session = Depends(get_db)):
    """Return recent prediction history from SQLite."""
    rows = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
    return [
        HistoryItem(
            id=r.id,
            input_features=json.loads(r.input_features),
            prediction=r.prediction,
            confidence=r.confidence,
            trend_adjusted=r.trend_adjusted,
            timestamp=r.timestamp,
        )
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# GET /meta
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/meta", response_model=MetaResponse, tags=["Data"])
async def meta():
    """Return valid categories, content ratings, model metrics, and trend data."""
    svc = get_model_service()
    trends = get_all_category_trends()
    return MetaResponse(
        categories=svc.categories,
        content_ratings=svc.content_ratings,
        model_metrics=svc.metrics,
        category_trends=trends,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "RateIQ API", "version": "3.0.0"}


# ══════════════════════════════════════════════════════════════════════════════
# GET /feature-importance
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Insights"])
async def feature_importance():
    """
    Return ranked feature importance from trained model.
    Includes both model importance and mutual information scores.
    """
    try:
        svc = get_model_service()
        ranked = svc.get_feature_importance_ranked()
        return FeatureImportanceResponse(
            items=[FeatureImportanceItem(**item) for item in ranked],
            model_type=svc.metrics.get("model_type", "unknown"),
        )
    except Exception as e:
        logger.exception("Feature importance error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GET /dataset-insights
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/dataset-insights", response_model=DatasetInsightsResponse, tags=["Insights"])
async def dataset_insights():
    """
    Automatically mine the loaded dataset for significant patterns.
    Returns insights like 'High-install apps rate higher' with statistics.
    """
    try:
        import pandas as pd
        import os
        # Resolve from project root regardless of working directory
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, "..", ".."))
        data_path = os.path.join(_root, "data", "apps.csv")
        df = None
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, low_memory=False)
            # Normalise column names from real dataset
            col_map = {
                "Rating": "rating", "Category": "category",
                "Size": "size_mb", "Installs": "installs", "Price": "price",
                "Reviews": "reviews", "is_free": "is_free",
                "Content Rating": "content_rating",
                "log_reviews": "log_reviews", "log_installs": "log_installs",
            }
            df = df.rename(columns=col_map)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            # Parse installs if still string
            if "installs" in df.columns and df["installs"].dtype == object:
                df["installs"] = df["installs"].astype(str).str.replace(",","").str.replace("+","").str.extract(r"([\d]+)")[0].astype(float).fillna(0)
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df = df[df["rating"].between(1.0, 5.0)].dropna(subset=["rating"])

        insights = generate_dataset_insights(df)
        rows = len(df) if df is not None else 0
        return DatasetInsightsResponse(
            insights=[DatasetInsightItem(**i) for i in insights],
            dataset_rows=rows,
            dataset_source="apps.csv" if df is not None else "unavailable",
        )
    except Exception as e:
        logger.exception("Dataset insights error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
