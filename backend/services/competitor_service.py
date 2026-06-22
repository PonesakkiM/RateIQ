"""
RateIQ – Competitor Gap Analyzer Service
Finds similar apps using cosine similarity and produces actionable gap analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("rateiq.competitor")

# ── Category average stats (derived from synthetic dataset, used as "market" data) ──
CATEGORY_BENCHMARKS: Dict[str, Dict] = {
    "Education":          {"avg_rating": 4.25, "avg_size_mb": 32.0, "pct_free": 0.82, "avg_installs": 250_000, "avg_reviews": 8_500},
    "Health & Fitness":   {"avg_rating": 4.18, "avg_size_mb": 45.0, "pct_free": 0.76, "avg_installs": 180_000, "avg_reviews": 6_200},
    "Productivity":       {"avg_rating": 4.10, "avg_size_mb": 28.0, "pct_free": 0.70, "avg_installs": 320_000, "avg_reviews": 12_000},
    "Tools":              {"avg_rating": 4.05, "avg_size_mb": 22.0, "pct_free": 0.78, "avg_installs": 500_000, "avg_reviews": 15_000},
    "Entertainment":      {"avg_rating": 3.95, "avg_size_mb": 55.0, "pct_free": 0.88, "avg_installs": 750_000, "avg_reviews": 22_000},
    "Social":             {"avg_rating": 3.90, "avg_size_mb": 60.0, "pct_free": 0.92, "avg_installs": 1_200_000, "avg_reviews": 45_000},
    "Shopping":           {"avg_rating": 3.88, "avg_size_mb": 38.0, "pct_free": 0.95, "avg_installs": 800_000, "avg_reviews": 18_000},
    "Game":               {"avg_rating": 3.82, "avg_size_mb": 85.0, "pct_free": 0.90, "avg_installs": 1_500_000, "avg_reviews": 60_000},
    "Travel & Local":     {"avg_rating": 4.00, "avg_size_mb": 42.0, "pct_free": 0.80, "avg_installs": 400_000, "avg_reviews": 10_000},
    "Communication":      {"avg_rating": 3.78, "avg_size_mb": 50.0, "pct_free": 0.88, "avg_installs": 2_000_000, "avg_reviews": 80_000},
    "Finance":            {"avg_rating": 3.85, "avg_size_mb": 25.0, "pct_free": 0.72, "avg_installs": 350_000, "avg_reviews": 9_000},
    "Music & Audio":      {"avg_rating": 4.02, "avg_size_mb": 48.0, "pct_free": 0.82, "avg_installs": 900_000, "avg_reviews": 30_000},
    "Photography":        {"avg_rating": 4.08, "avg_size_mb": 52.0, "pct_free": 0.85, "avg_installs": 600_000, "avg_reviews": 20_000},
    "Maps & Navigation":  {"avg_rating": 4.12, "avg_size_mb": 65.0, "pct_free": 0.88, "avg_installs": 700_000, "avg_reviews": 25_000},
    "News & Magazines":   {"avg_rating": 3.96, "avg_size_mb": 30.0, "pct_free": 0.75, "avg_installs": 300_000, "avg_reviews": 7_000},
    "Sports":             {"avg_rating": 3.92, "avg_size_mb": 40.0, "pct_free": 0.84, "avg_installs": 450_000, "avg_reviews": 14_000},
    "Business":           {"avg_rating": 3.80, "avg_size_mb": 35.0, "pct_free": 0.65, "avg_installs": 150_000, "avg_reviews": 4_500},
    "Lifestyle":          {"avg_rating": 4.00, "avg_size_mb": 33.0, "pct_free": 0.80, "avg_installs": 220_000, "avg_reviews": 7_200},
    "Parenting":          {"avg_rating": 4.15, "avg_size_mb": 28.0, "pct_free": 0.78, "avg_installs": 120_000, "avg_reviews": 3_500},
    "Food & Drink":       {"avg_rating": 4.05, "avg_size_mb": 30.0, "pct_free": 0.82, "avg_installs": 280_000, "avg_reviews": 9_500},
    "Medical":            {"avg_rating": 4.20, "avg_size_mb": 25.0, "pct_free": 0.68, "avg_installs": 80_000,  "avg_reviews": 2_200},
    "Dating":             {"avg_rating": 3.70, "avg_size_mb": 45.0, "pct_free": 0.80, "avg_installs": 500_000, "avg_reviews": 18_000},
    "Art & Design":       {"avg_rating": 4.12, "avg_size_mb": 38.0, "pct_free": 0.83, "avg_installs": 200_000, "avg_reviews": 6_800},
    "Video Players":      {"avg_rating": 4.05, "avg_size_mb": 55.0, "pct_free": 0.86, "avg_installs": 800_000, "avg_reviews": 28_000},
    "Weather":            {"avg_rating": 4.08, "avg_size_mb": 20.0, "pct_free": 0.78, "avg_installs": 400_000, "avg_reviews": 11_000},
}

# Default fallback
DEFAULT_BENCHMARK = {"avg_rating": 4.00, "avg_size_mb": 40.0, "pct_free": 0.80, "avg_installs": 300_000, "avg_reviews": 10_000}

# Simulated competitor pool (top 5 per category) — representative archetypes
COMPETITOR_ARCHETYPES: Dict[str, List[Dict]] = {
    "Education": [
        {"name": "LearnPro",  "rating": 4.5, "size_mb": 28.0, "price": 0.0, "installs": 1_000_000, "content_rating": "Everyone", "reviews": 35_000},
        {"name": "StudyBuddy","rating": 4.3, "size_mb": 35.0, "price": 0.0, "installs": 500_000,   "content_rating": "Everyone", "reviews": 18_000},
        {"name": "MathGenius","rating": 4.2, "size_mb": 22.0, "price": 1.99,"installs": 200_000,   "content_rating": "Everyone", "reviews": 8_000},
        {"name": "CodeKids",  "rating": 4.4, "size_mb": 40.0, "price": 0.0, "installs": 750_000,   "content_rating": "Everyone", "reviews": 25_000},
        {"name": "QuizMaster","rating": 4.1, "size_mb": 18.0, "price": 0.0, "installs": 300_000,   "content_rating": "Everyone", "reviews": 10_000},
    ],
    "Game": [
        {"name": "PuzzleKing","rating": 4.2, "size_mb": 75.0, "price": 0.0, "installs": 5_000_000, "content_rating": "Everyone","reviews": 180_000},
        {"name": "SpeedRacer","rating": 4.0, "size_mb": 120.0,"price": 0.0, "installs": 2_000_000, "content_rating": "Everyone","reviews": 75_000},
        {"name": "StrategyX", "rating": 3.9, "size_mb": 90.0, "price": 2.99,"installs": 500_000,  "content_rating": "Teen",    "reviews": 20_000},
        {"name": "CasualFun", "rating": 4.3, "size_mb": 60.0, "price": 0.0, "installs": 10_000_000,"content_rating": "Everyone","reviews": 350_000},
        {"name": "RPGQuest",  "rating": 4.1, "size_mb": 150.0,"price": 0.0, "installs": 1_000_000, "content_rating": "Teen",    "reviews": 40_000},
    ],
    "Productivity": [
        {"name": "TaskFlow",  "rating": 4.4, "size_mb": 25.0, "price": 0.0, "installs": 2_000_000, "content_rating": "Everyone","reviews": 65_000},
        {"name": "NoteSync",  "rating": 4.3, "size_mb": 30.0, "price": 0.0, "installs": 1_500_000, "content_rating": "Everyone","reviews": 50_000},
        {"name": "FocusTimer","rating": 4.5, "size_mb": 15.0, "price": 0.0, "installs": 800_000,   "content_rating": "Everyone","reviews": 28_000},
        {"name": "DocEdit",   "rating": 4.1, "size_mb": 45.0, "price": 4.99,"installs": 300_000,   "content_rating": "Everyone","reviews": 9_000},
        {"name": "MindMap",   "rating": 4.0, "size_mb": 35.0, "price": 0.0, "installs": 600_000,   "content_rating": "Everyone","reviews": 20_000},
    ],
    "Health & Fitness": [
        {"name": "FitTrack",  "rating": 4.5, "size_mb": 38.0, "price": 0.0, "installs": 3_000_000, "content_rating": "Everyone","reviews": 90_000},
        {"name": "YogaFlow",  "rating": 4.4, "size_mb": 50.0, "price": 0.0, "installs": 800_000,   "content_rating": "Everyone","reviews": 30_000},
        {"name": "NutriPlan", "rating": 4.3, "size_mb": 35.0, "price": 2.99,"installs": 400_000,   "content_rating": "Everyone","reviews": 12_000},
        {"name": "RunMate",   "rating": 4.2, "size_mb": 42.0, "price": 0.0, "installs": 600_000,   "content_rating": "Everyone","reviews": 22_000},
        {"name": "SleepWell", "rating": 4.6, "size_mb": 28.0, "price": 0.0, "installs": 1_200_000, "content_rating": "Everyone","reviews": 45_000},
    ],
    "Tools": [
        {"name": "CleanMaster","rating": 4.1,"size_mb": 20.0, "price": 0.0, "installs": 5_000_000, "content_rating": "Everyone","reviews": 200_000},
        {"name": "WiFiScan",  "rating": 4.3, "size_mb": 12.0, "price": 0.0, "installs": 1_000_000, "content_rating": "Everyone","reviews": 35_000},
        {"name": "FileMan",   "rating": 4.4, "size_mb": 18.0, "price": 0.0, "installs": 2_000_000, "content_rating": "Everyone","reviews": 70_000},
        {"name": "QRPro",     "rating": 4.2, "size_mb": 10.0, "price": 0.0, "installs": 3_000_000, "content_rating": "Everyone","reviews": 95_000},
        {"name": "VaultKey",  "rating": 4.5, "size_mb": 15.0, "price": 1.99,"installs": 500_000,   "content_rating": "Everyone","reviews": 18_000},
    ],
    "Entertainment": [
        {"name": "StreamNow", "rating": 4.2, "size_mb": 65.0, "price": 0.0, "installs": 5_000_000, "content_rating": "Teen",   "reviews": 180_000},
        {"name": "TikVibe",   "rating": 3.9, "size_mb": 80.0, "price": 0.0, "installs": 10_000_000,"content_rating": "Teen",   "reviews": 350_000},
        {"name": "PodCast+",  "rating": 4.4, "size_mb": 40.0, "price": 0.0, "installs": 1_000_000, "content_rating": "Everyone","reviews": 42_000},
        {"name": "AniWatch",  "rating": 4.1, "size_mb": 55.0, "price": 0.0, "installs": 2_000_000, "content_rating": "Teen",   "reviews": 65_000},
        {"name": "LiveFest",  "rating": 3.8, "size_mb": 70.0, "price": 2.99,"installs": 300_000,   "content_rating": "Teen",   "reviews": 10_000},
    ],
}
# Add generic fallback for all other categories
for _cat in CATEGORY_BENCHMARKS:
    if _cat not in COMPETITOR_ARCHETYPES:
        COMPETITOR_ARCHETYPES[_cat] = [
            {"name": f"{_cat} Leader",  "rating": CATEGORY_BENCHMARKS[_cat]["avg_rating"] + 0.3,
             "size_mb": CATEGORY_BENCHMARKS[_cat]["avg_size_mb"] * 0.9,
             "price": 0.0 if CATEGORY_BENCHMARKS[_cat]["pct_free"] > 0.75 else 1.99,
             "installs": int(CATEGORY_BENCHMARKS[_cat]["avg_installs"] * 3),
             "content_rating": "Everyone",
             "reviews": int(CATEGORY_BENCHMARKS[_cat]["avg_reviews"] * 2)},
            {"name": f"{_cat} Pro",     "rating": CATEGORY_BENCHMARKS[_cat]["avg_rating"] + 0.1,
             "size_mb": CATEGORY_BENCHMARKS[_cat]["avg_size_mb"],
             "price": 0.0,
             "installs": int(CATEGORY_BENCHMARKS[_cat]["avg_installs"] * 1.5),
             "content_rating": "Everyone",
             "reviews": int(CATEGORY_BENCHMARKS[_cat]["avg_reviews"] * 1.2)},
            {"name": f"{_cat} Basic",   "rating": CATEGORY_BENCHMARKS[_cat]["avg_rating"] - 0.1,
             "size_mb": CATEGORY_BENCHMARKS[_cat]["avg_size_mb"] * 0.7,
             "price": 0.0,
             "installs": int(CATEGORY_BENCHMARKS[_cat]["avg_installs"] * 0.5),
             "content_rating": "Everyone",
             "reviews": int(CATEGORY_BENCHMARKS[_cat]["avg_reviews"] * 0.5)},
        ]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _app_to_vector(app: dict) -> np.ndarray:
    """Convert app metadata to numerical feature vector for similarity."""
    return np.array([
        np.log1p(app.get("size_mb", 30)),
        np.log1p(app.get("installs", 100_000)),
        float(app.get("price", 0.0)),
        np.log1p(app.get("reviews", 1_000)),
    ], dtype=float)


def _rating_efficiency(rating: float, installs: int) -> float:
    """Rating per log-install unit — measures how efficiently an app converts installs to stars."""
    log_inst = np.log1p(installs)
    return round(rating / log_inst if log_inst > 0 else 0.0, 4)


def analyze_competitors(app_data: dict) -> Dict[str, Any]:
    """
    Full competitor gap analysis for a given app.
    Returns similarity scores, benchmark comparisons, feature gaps, and insights.
    """
    category = app_data.get("category", "Tools")
    benchmark = CATEGORY_BENCHMARKS.get(category, DEFAULT_BENCHMARK)
    competitors = COMPETITOR_ARCHETYPES.get(category, COMPETITOR_ARCHETYPES.get("Tools", []))

    # ── Similarity scoring ─────────────────────────────────────────────────
    user_vec = _app_to_vector(app_data)
    similar_apps = []
    for comp in competitors:
        comp_vec = _app_to_vector(comp)
        sim_score = _cosine_similarity(user_vec, comp_vec)
        gap = {
            "name":             comp["name"],
            "similarity_score": round(sim_score * 100, 1),
            "rating":           comp["rating"],
            "size_mb":          comp["size_mb"],
            "price":            comp["price"],
            "installs":         comp["installs"],
            "reviews":          comp["reviews"],
            "rating_efficiency": _rating_efficiency(comp["rating"], comp["installs"]),
        }
        similar_apps.append(gap)

    similar_apps.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_3 = similar_apps[:3]

    # ── Feature gap computation ────────────────────────────────────────────
    user_rating    = app_data.get("predicted_rating", 3.8)
    user_size      = app_data.get("size_mb", 40)
    user_installs  = app_data.get("installs", 100_000)
    user_price     = app_data.get("price", 0.0)
    user_reviews   = app_data.get("reviews", 1_000)
    user_is_free   = app_data.get("is_free", 1)

    avg_comp_size     = np.mean([c["size_mb"] for c in top_3]) if top_3 else benchmark["avg_size_mb"]
    avg_comp_rating   = np.mean([c["rating"] for c in top_3]) if top_3 else benchmark["avg_rating"]
    avg_comp_installs = np.mean([c["installs"] for c in top_3]) if top_3 else benchmark["avg_installs"]
    avg_comp_price    = np.mean([c["price"] for c in top_3]) if top_3 else 0.0
    avg_comp_reviews  = np.mean([c["reviews"] for c in top_3]) if top_3 else benchmark["avg_reviews"]

    feature_gaps = []
    performance_gaps = []
    insights = []

    # Size gap
    size_pct = ((user_size - avg_comp_size) / avg_comp_size * 100) if avg_comp_size > 0 else 0
    if abs(size_pct) > 15:
        direction = "larger" if size_pct > 0 else "smaller"
        impact    = "negative" if size_pct > 0 else "positive"
        feature_gaps.append({
            "feature": "App Size",
            "your_value": f"{user_size:.1f} MB",
            "competitor_avg": f"{avg_comp_size:.1f} MB",
            "gap_pct": round(size_pct, 1),
            "impact": impact,
            "message": f"Your app is {abs(size_pct):.0f}% {direction} than top competitors → {impact} impact on rating",
        })
        if size_pct > 20:
            insights.append(f"📦 App size is {size_pct:.0f}% above category average. Reducing APK size improves install rates and indirectly boosts ratings.")

    # Price gap
    top_free_pct = benchmark["pct_free"]
    if user_price > 0 and top_free_pct > 0.75:
        feature_gaps.append({
            "feature": "Pricing",
            "your_value": f"${user_price:.2f}",
            "competitor_avg": f"{top_free_pct*100:.0f}% are free",
            "gap_pct": None,
            "impact": "negative",
            "message": f"Top apps in {category} are {top_free_pct*100:.0f}% free → pricing mismatch detected",
        })
        insights.append(f"💰 {top_free_pct*100:.0f}% of top {category} apps are free. Consider a freemium model or trial period.")

    # Install gap
    install_gap_pct = ((avg_comp_installs - user_installs) / avg_comp_installs * 100) if avg_comp_installs > 0 else 0
    if install_gap_pct > 50:
        feature_gaps.append({
            "feature": "Install Count",
            "your_value": f"{user_installs:,}",
            "competitor_avg": f"{avg_comp_installs:,.0f}",
            "gap_pct": round(-install_gap_pct, 1),
            "impact": "negative",
            "message": f"Your installs are {install_gap_pct:.0f}% below top competitors. Low social proof impacts perceived quality.",
        })
        insights.append("📥 Low install count signals low adoption. Prioritize ASO (App Store Optimization) to improve discovery.")

    # Review gap
    review_gap_pct = ((avg_comp_reviews - user_reviews) / avg_comp_reviews * 100) if avg_comp_reviews > 0 else 0
    if review_gap_pct > 40:
        feature_gaps.append({
            "feature": "Review Count",
            "your_value": f"{user_reviews:,}",
            "competitor_avg": f"{avg_comp_reviews:,.0f}",
            "gap_pct": round(-review_gap_pct, 1),
            "impact": "negative",
            "message": f"Your review count is {review_gap_pct:.0f}% below the top 3 similar apps.",
        })
        insights.append("⭐ Use in-app review prompts after positive interactions. Reviews are the #1 social proof signal.")

    # Rating gap
    rating_gap = avg_comp_rating - user_rating
    if rating_gap > 0.1:
        performance_gaps.append({
            "metric": "Predicted Rating",
            "your_value": round(user_rating, 2),
            "competitor_avg": round(avg_comp_rating, 2),
            "gap": round(rating_gap, 2),
            "message": f"Top competitors rate {rating_gap:.2f} stars higher than your predicted {user_rating:.2f}",
        })

    # Rating efficiency
    user_eff = _rating_efficiency(user_rating, user_installs)
    comp_eff = np.mean([c["rating_efficiency"] for c in top_3]) if top_3 else _rating_efficiency(avg_comp_rating, int(avg_comp_installs))
    eff_gap = comp_eff - user_eff
    if abs(eff_gap) > 0.001:
        direction = "lower" if eff_gap > 0 else "higher"
        performance_gaps.append({
            "metric": "Rating Efficiency",
            "your_value": round(user_eff, 4),
            "competitor_avg": round(comp_eff, 4),
            "gap": round(eff_gap, 4),
            "message": f"Your rating efficiency is {direction} than competitors — you're getting {direction} stars per install unit",
        })

    if not insights:
        insights.append(f"✅ Your app is well-positioned vs. {category} competitors. Focus on maintaining update cadence and review quality.")

    return {
        "category": category,
        "benchmark": benchmark,
        "similar_apps": similar_apps,
        "top_competitors": top_3,
        "feature_gaps": feature_gaps,
        "performance_gaps": performance_gaps,
        "insights": insights,
        "summary": {
            "avg_competitor_rating": round(avg_comp_rating, 2),
            "avg_competitor_size_mb": round(avg_comp_size, 1),
            "avg_competitor_installs": int(avg_comp_installs),
            "avg_competitor_reviews": int(avg_comp_reviews),
            "market_free_pct": round(benchmark["pct_free"] * 100, 1),
            "your_rating": round(user_rating, 2),
            "total_gaps_found": len(feature_gaps) + len(performance_gaps),
        },
    }
