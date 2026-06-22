"""
RateIQ – Trend-Aware Rating Booster Service
Adjusts ML predictions using category market saturation, popularity trends,
and competition level data.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("rateiq.trend")

# ── Category Trend Profiles ───────────────────────────────────────────────────
# trend_score: -1.0 (highly saturated) to +1.0 (emerging/growing)
# saturation: 0.0 (low) to 1.0 (very saturated)
# competition_level: "Low" | "Medium" | "High" | "Very High"
# popularity_index: 0-100 (how popular the category is overall)
CATEGORY_TRENDS: Dict[str, Dict] = {
    "Education":          {"trend_score": +0.35, "saturation": 0.40, "competition_level": "Medium",    "popularity_index": 72, "yoy_growth": "+18%", "market_stage": "Growing"},
    "Health & Fitness":   {"trend_score": +0.42, "saturation": 0.35, "competition_level": "Medium",    "popularity_index": 78, "yoy_growth": "+22%", "market_stage": "Growing"},
    "Productivity":       {"trend_score": +0.20, "saturation": 0.55, "competition_level": "High",      "popularity_index": 68, "yoy_growth": "+12%", "market_stage": "Mature"},
    "Tools":              {"trend_score": +0.10, "saturation": 0.65, "competition_level": "High",      "popularity_index": 75, "yoy_growth": "+8%",  "market_stage": "Mature"},
    "Game":               {"trend_score": -0.25, "saturation": 0.90, "competition_level": "Very High", "popularity_index": 95, "yoy_growth": "+5%",  "market_stage": "Saturated"},
    "Entertainment":      {"trend_score": -0.15, "saturation": 0.80, "competition_level": "Very High", "popularity_index": 90, "yoy_growth": "+6%",  "market_stage": "Saturated"},
    "Social":             {"trend_score": -0.30, "saturation": 0.85, "competition_level": "Very High", "popularity_index": 88, "yoy_growth": "+3%",  "market_stage": "Saturated"},
    "Shopping":           {"trend_score": +0.05, "saturation": 0.70, "competition_level": "High",      "popularity_index": 80, "yoy_growth": "+10%", "market_stage": "Mature"},
    "Travel & Local":     {"trend_score": +0.30, "saturation": 0.45, "competition_level": "Medium",    "popularity_index": 65, "yoy_growth": "+20%", "market_stage": "Growing"},
    "Communication":      {"trend_score": -0.20, "saturation": 0.78, "competition_level": "Very High", "popularity_index": 85, "yoy_growth": "+4%",  "market_stage": "Saturated"},
    "Finance":            {"trend_score": +0.38, "saturation": 0.42, "competition_level": "Medium",    "popularity_index": 70, "yoy_growth": "+25%", "market_stage": "Growing"},
    "Music & Audio":      {"trend_score": +0.08, "saturation": 0.68, "competition_level": "High",      "popularity_index": 76, "yoy_growth": "+9%",  "market_stage": "Mature"},
    "Photography":        {"trend_score": +0.15, "saturation": 0.60, "competition_level": "High",      "popularity_index": 73, "yoy_growth": "+11%", "market_stage": "Mature"},
    "Maps & Navigation":  {"trend_score": +0.12, "saturation": 0.55, "competition_level": "High",      "popularity_index": 69, "yoy_growth": "+14%", "market_stage": "Mature"},
    "News & Magazines":   {"trend_score": -0.10, "saturation": 0.72, "competition_level": "High",      "popularity_index": 65, "yoy_growth": "+2%",  "market_stage": "Declining"},
    "Sports":             {"trend_score": +0.22, "saturation": 0.50, "competition_level": "Medium",    "popularity_index": 70, "yoy_growth": "+16%", "market_stage": "Growing"},
    "Business":           {"trend_score": +0.25, "saturation": 0.45, "competition_level": "Medium",    "popularity_index": 60, "yoy_growth": "+15%", "market_stage": "Growing"},
    "Lifestyle":          {"trend_score": +0.18, "saturation": 0.52, "competition_level": "Medium",    "popularity_index": 67, "yoy_growth": "+13%", "market_stage": "Growing"},
    "Parenting":          {"trend_score": +0.45, "saturation": 0.30, "competition_level": "Low",       "popularity_index": 55, "yoy_growth": "+28%", "market_stage": "Emerging"},
    "Food & Drink":       {"trend_score": +0.28, "saturation": 0.48, "competition_level": "Medium",    "popularity_index": 71, "yoy_growth": "+17%", "market_stage": "Growing"},
    "Medical":            {"trend_score": +0.40, "saturation": 0.32, "competition_level": "Low",       "popularity_index": 58, "yoy_growth": "+30%", "market_stage": "Emerging"},
    "Dating":             {"trend_score": -0.35, "saturation": 0.88, "competition_level": "Very High", "popularity_index": 78, "yoy_growth": "+1%",  "market_stage": "Saturated"},
    "Art & Design":       {"trend_score": +0.32, "saturation": 0.38, "competition_level": "Low",       "popularity_index": 60, "yoy_growth": "+20%", "market_stage": "Growing"},
    "Video Players":      {"trend_score": -0.05, "saturation": 0.75, "competition_level": "High",      "popularity_index": 80, "yoy_growth": "+7%",  "market_stage": "Mature"},
    "Weather":            {"trend_score": +0.05, "saturation": 0.62, "competition_level": "Medium",    "popularity_index": 65, "yoy_growth": "+8%",  "market_stage": "Mature"},
    "Personalization":    {"trend_score": +0.10, "saturation": 0.65, "competition_level": "High",      "popularity_index": 68, "yoy_growth": "+9%",  "market_stage": "Mature"},
    "Libraries & Demo":   {"trend_score": +0.02, "saturation": 0.55, "competition_level": "Low",       "popularity_index": 40, "yoy_growth": "+5%",  "market_stage": "Niche"},
    "Comics":             {"trend_score": +0.28, "saturation": 0.40, "competition_level": "Low",       "popularity_index": 52, "yoy_growth": "+18%", "market_stage": "Growing"},
    "Auto & Vehicles":    {"trend_score": +0.15, "saturation": 0.42, "competition_level": "Low",       "popularity_index": 55, "yoy_growth": "+12%", "market_stage": "Growing"},
    "Beauty":             {"trend_score": +0.35, "saturation": 0.38, "competition_level": "Low",       "popularity_index": 62, "yoy_growth": "+22%", "market_stage": "Growing"},
    "Books & Reference":  {"trend_score": +0.12, "saturation": 0.55, "competition_level": "Medium",    "popularity_index": 60, "yoy_growth": "+10%", "market_stage": "Mature"},
    "Events":             {"trend_score": +0.20, "saturation": 0.42, "competition_level": "Low",       "popularity_index": 55, "yoy_growth": "+15%", "market_stage": "Growing"},
    "House & Home":       {"trend_score": +0.38, "saturation": 0.35, "competition_level": "Low",       "popularity_index": 60, "yoy_growth": "+24%", "market_stage": "Emerging"},
}

DEFAULT_TREND = {"trend_score": 0.0, "saturation": 0.55, "competition_level": "Medium", "popularity_index": 65, "yoy_growth": "+10%", "market_stage": "Mature"}

# Trend adjustment weights
TREND_WEIGHT     = 0.12   # max ±0.12 stars from trend
SAT_WEIGHT       = 0.08   # max -0.08 for full saturation
COMP_PENALTIES   = {"Low": 0.0, "Medium": -0.02, "High": -0.05, "Very High": -0.10}


def compute_trend_adjustment(category: str, base_prediction: float) -> Dict[str, Any]:
    """
    Compute trend-adjusted rating given category and base ML prediction.
    
    Returns:
        trend_profile: raw trend data for the category
        adjustment:    stars added/removed by trend
        adjusted_rating: final trend-boosted rating
        explanation: human-readable explanation
    """
    trend = CATEGORY_TRENDS.get(category, DEFAULT_TREND)

    trend_adj  = trend["trend_score"] * TREND_WEIGHT
    sat_adj    = -trend["saturation"] * SAT_WEIGHT
    comp_adj   = COMP_PENALTIES.get(trend["competition_level"], -0.02)

    total_adjustment = round(trend_adj + sat_adj + comp_adj, 3)
    adjusted_rating  = round(max(1.0, min(5.0, base_prediction + total_adjustment)), 2)

    # ── Build explanation ─────────────────────────────────────────────────────
    parts = []
    if trend["trend_score"] > 0.2:
        parts.append(f"{category} is a **growing market** ({trend['yoy_growth']} YoY) → trend boost of +{trend_adj:.3f} ★")
    elif trend["trend_score"] < -0.1:
        parts.append(f"{category} market is **saturated/declining** → trend penalty of {trend_adj:.3f} ★")
    else:
        parts.append(f"{category} market is **stable** → minimal trend adjustment ({trend_adj:+.3f} ★)")

    if trend["saturation"] > 0.70:
        parts.append(f"High category saturation ({trend['saturation']*100:.0f}%) → competition penalty of {sat_adj:.3f} ★")
    elif trend["saturation"] < 0.40:
        parts.append(f"Low saturation ({trend['saturation']*100:.0f}%) — less competition in this space → favorable conditions")

    if trend["competition_level"] in ("High", "Very High"):
        parts.append(f"{trend['competition_level']} competition level → additional adjustment of {comp_adj:.3f} ★")

    explanation = " | ".join(parts)

    # ── Stage-based advice ────────────────────────────────────────────────────
    stage_advice = {
        "Emerging":  "🌱 This is an emerging category — early movers capture disproportionate ratings. Ship fast and iterate.",
        "Growing":   "📈 Growing category with increasing user demand. Quality and frequent updates will give you an edge.",
        "Mature":    "⚖️ Mature market. Differentiation through UX polish and niche targeting is key to standing out.",
        "Saturated": "🔴 Highly saturated space. Very hard to get above-average ratings — consider a unique value proposition.",
        "Declining": "📉 Declining category engagement. Focus on retaining existing users and consider pivoting your category tag.",
        "Niche":     "🎯 Niche category with low competition. Strong opportunity if you serve the audience well.",
    }
    stage_msg = stage_advice.get(trend["market_stage"], "")

    return {
        "category":         category,
        "trend_profile":    trend,
        "base_prediction":  round(base_prediction, 2),
        "trend_adjustment": total_adjustment,
        "adjusted_rating":  adjusted_rating,
        "adjustment_breakdown": {
            "trend_component":       round(trend_adj, 3),
            "saturation_component":  round(sat_adj, 3),
            "competition_component": round(comp_adj, 3),
        },
        "explanation":      explanation,
        "stage_advice":     stage_msg,
        "market_stage":     trend["market_stage"],
        "competition_level": trend["competition_level"],
        "yoy_growth":       trend["yoy_growth"],
        "popularity_index": trend["popularity_index"],
    }


def get_all_category_trends() -> Dict[str, Dict]:
    """Return full trend data for all categories (used in frontend visualizations)."""
    result = {}
    for cat, trend in CATEGORY_TRENDS.items():
        result[cat] = {
            **trend,
            "example_adjustment": compute_trend_adjustment(cat, 4.0)["trend_adjustment"],
        }
    return result
