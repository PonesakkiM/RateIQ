"""
RateIQ – Pydantic Schemas (request / response models)
Covers: prediction, chat, competitor analysis, trend boost.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


# ── Shared app metadata (used across multiple endpoints) ─────────────────────

class AppMetadata(BaseModel):
    category: str          = Field(..., example="Education")
    size_mb: float         = Field(..., ge=0.1, le=2000, example=25.5)
    installs: int          = Field(..., ge=0, example=100000)
    price: float           = Field(0.0, ge=0.0, le=999.99, example=0.0)
    content_rating: str    = Field(..., example="Everyone")
    reviews: int           = Field(..., ge=0, example=5000)
    update_days: int       = Field(..., ge=0, le=3650, example=30)
    num_screenshots: int   = Field(3, ge=0, le=8, example=3)
    has_ads: int           = Field(0, ge=0, le=1, example=0)
    is_free: int           = Field(1, ge=0, le=1, example=1)

    @field_validator("category")
    @classmethod
    def category_not_empty(cls, v):
        if not v.strip():
            raise ValueError("category must not be empty")
        return v.strip()

    @field_validator("content_rating")
    @classmethod
    def cr_not_empty(cls, v):
        if not v.strip():
            raise ValueError("content_rating must not be empty")
        return v.strip()

    def to_dict(self) -> dict:
        return self.model_dump()


# ── /predict ─────────────────────────────────────────────────────────────────

class PredictRequest(AppMetadata):
    pass  # inherits all fields


class ShapValue(BaseModel):
    feature: str
    label: str
    value: float
    raw_feature_value: float


class TrendAdjustment(BaseModel):
    base_prediction: float
    trend_adjustment: float
    adjusted_rating: float
    market_stage: str
    competition_level: str
    yoy_growth: str
    popularity_index: int
    explanation: str
    stage_advice: str
    adjustment_breakdown: Dict[str, float]


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: float
    confidence: float
    confidence_tier: Optional[str] = None       # "High" / "Medium" / "Low" etc.
    confidence_desc: Optional[str] = None       # human-readable interpretation
    shap_values: List[ShapValue]
    model_metrics: dict
    recommendation: str
    trend: Optional[TrendAdjustment] = None


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, example="Why is my app rating low?")
    app_data: Optional[AppMetadata] = None
    prediction_data: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[ChatMessage]] = None


class IssueItem(BaseModel):
    severity: str
    feature: str
    issue: str
    root_cause: str
    fix: str
    expected_impact: Optional[str] = None


class RecommendationCard(BaseModel):
    title: str
    action: str
    impact: str
    severity: str


class ChatResponse(BaseModel):
    response: str
    detected_intents: List[str]
    issues_found: int
    issue_summary: List[IssueItem]
    recommendations: List[RecommendationCard]
    follow_up_questions: List[str]
    confidence: str


# ── /competitor-analysis ──────────────────────────────────────────────────────

class CompetitorRequest(BaseModel):
    app_data: AppMetadata
    predicted_rating: Optional[float] = Field(None, ge=1.0, le=5.0)


class FeatureGap(BaseModel):
    feature: str
    your_value: str
    competitor_avg: str
    gap_pct: Optional[float]
    impact: str
    message: str


class PerformanceGap(BaseModel):
    metric: str
    your_value: float
    competitor_avg: float
    gap: float
    message: str


class CompetitorApp(BaseModel):
    name: str
    similarity_score: float
    rating: float
    size_mb: float
    price: float
    installs: int
    reviews: int
    rating_efficiency: float


class CompetitorSummary(BaseModel):
    avg_competitor_rating: float
    avg_competitor_size_mb: float
    avg_competitor_installs: int
    avg_competitor_reviews: int
    market_free_pct: float
    your_rating: float
    total_gaps_found: int


class CompetitorResponse(BaseModel):
    category: str
    similar_apps: List[CompetitorApp]
    top_competitors: List[CompetitorApp]
    feature_gaps: List[FeatureGap]
    performance_gaps: List[PerformanceGap]
    insights: List[str]
    summary: CompetitorSummary


# ── /trend ───────────────────────────────────────────────────────────────────

class TrendRequest(BaseModel):
    category: str = Field(..., example="Game")
    base_prediction: float = Field(..., ge=1.0, le=5.0, example=3.9)


# ── History / Meta ────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: int
    input_features: dict
    prediction: float
    confidence: float
    trend_adjusted: Optional[float] = None
    timestamp: datetime


class MetaResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    categories: List[str]
    content_ratings: List[str]
    model_metrics: dict
    category_trends: Optional[Dict[str, Any]] = None


# ── Feature Importance ────────────────────────────────────────────────────────

class FeatureImportanceItem(BaseModel):
    feature: str
    label: str
    importance: float
    mi_score: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    items: List[FeatureImportanceItem]
    model_type: str


# ── Dataset Insights ──────────────────────────────────────────────────────────

class DatasetInsightItem(BaseModel):
    icon: str
    finding: str
    statistic: str
    implication: str
    category: str
    strength: float


class DatasetInsightsResponse(BaseModel):
    insights: List[DatasetInsightItem]
    dataset_rows: int
    dataset_source: str
