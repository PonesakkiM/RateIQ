<<<<<<< HEAD
# RateIQ
=======
# 🔮 RateIQ v2 — AI App Rating Optimization Platform

> **Full-stack AI SaaS for mobile app developers. Predict Play Store ratings, analyze competitors, get AI-driven advice, and leverage market trend data — all in one platform.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey?logo=sqlite)

---

## 📋 What RateIQ Does

RateIQ is a production-ready AI platform for mobile app developers to:

| Module | Description |
|---|---|
| 🔮 **Prediction Engine** | XGBoost model predicts Play Store rating from 10 metadata features |
| 📊 **Competitor Gap Analyzer** | Finds similar apps using cosine similarity, surfaces feature + performance gaps |
| 💬 **AI Advisor** | Natural language chatbot acting as AI Product Manager |
| 🌍 **Trend Booster** | Adjusts predictions using category market saturation and competition data |
| 🔬 **What-If Analysis** | Real-time prediction delta as you adjust individual features |
| 🔗 **URL Auto-fill** | Simulates Play Store metadata extraction from app URL |

---

## 📂 Project Structure

```
rateiq/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── core/config.py             # Settings (env-aware)
│   ├── api/
│   │   ├── routes.py              # All API endpoints
│   │   └── schemas.py             # Pydantic request/response models
│   ├── db/database.py             # SQLAlchemy ORM (3 tables)
│   ├── services/
│   │   ├── model_service.py       # XGBoost + SHAP inference
│   │   ├── competitor_service.py  # Cosine similarity gap analyzer
│   │   ├── advisor_service.py     # NLP AI advisor engine
│   │   └── trend_service.py       # Category trend engine
│   ├── models/
│   │   ├── train_model.py         # Training script
│   │   └── model_artifacts.pkl    # Trained artifacts (generated)
│   └── requirements.txt
├── frontend/
│   ├── app.py                     # Streamlit dashboard (7 pages)
│   ├── api_client.py              # REST client
│   ├── styles.py                  # Dark/Light CSS themes
│   ├── requirements.txt
│   └── .streamlit/config.toml
├── data/
│   └── apps.csv                   # 6,000-app synthetic dataset (generated)
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── notebooks/
│   └── RateIQ_Analysis.ipynb
└── README.md
```

---

## ⚡ Quick Start (Local)

### Prerequisites
- Python 3.11+

### Step 1 — Install dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

### Step 2 — Train the ML model

```bash
cd backend/models
python train_model.py
cd ../..
```

This generates:
- `backend/models/model_artifacts.pkl` — trained XGBoost model + SHAP explainer
- `data/apps.csv` — 6,000-app synthetic dataset

### Step 3 — Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Step 4 — Start the frontend

```bash
streamlit run frontend/app.py
```

Dashboard: http://localhost:8501

---

## 🐳 Docker Deployment

```bash
# First, train model locally
cd backend/models && python train_model.py && cd ../..

# Build and run all services
cd docker
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

---

## 📡 API Reference

### POST `/api/v1/predict`
Predict app rating with SHAP explanation and trend adjustment.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Education",
    "size_mb": 25.5,
    "installs": 100000,
    "price": 0.0,
    "content_rating": "Everyone",
    "reviews": 5000,
    "update_days": 30,
    "num_screenshots": 5,
    "has_ads": 0,
    "is_free": 1
  }'
```

**Response:**
```json
{
  "prediction": 4.15,
  "confidence": 0.78,
  "shap_values": [
    {"feature": "log_reviews", "label": "Review Count", "value": 0.142, "raw_feature_value": 8.52},
    {"feature": "update_days", "label": "Days Since Update", "value": -0.089, "raw_feature_value": 30}
  ],
  "model_metrics": {"mae": 0.22, "r2": 0.30, "rmse": 0.29, "model_type": "XGBoost"},
  "recommendation": "Regular updates signal quality. Encourage user reviews after task completion.",
  "trend": {
    "base_prediction": 4.15,
    "trend_adjustment": 0.026,
    "adjusted_rating": 4.18,
    "market_stage": "Growing",
    "competition_level": "Medium",
    "yoy_growth": "+18%",
    "explanation": "Education is a growing market (+18% YoY) → trend boost",
    "stage_advice": "📈 Growing category. Quality and updates give you an edge."
  }
}
```

### POST `/api/v1/chat`
Natural language AI advisor.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is my app rating low?",
    "app_data": {"category": "Game", "size_mb": 150, "installs": 10000, "price": 0.0,
                 "content_rating": "Everyone", "reviews": 50, "update_days": 300,
                 "num_screenshots": 2, "has_ads": 1, "is_free": 1}
  }'
```

### POST `/api/v1/competitor-analysis`
Competitor gap analysis with similarity scoring.

```bash
curl -X POST http://localhost:8000/api/v1/competitor-analysis \
  -H "Content-Type: application/json" \
  -d '{"app_data": {"category": "Game", "size_mb": 80, "installs": 50000,
                    "price": 0.0, "content_rating": "Everyone", "reviews": 1000,
                    "update_days": 60, "num_screenshots": 4, "has_ads": 1, "is_free": 1}}'
```

### POST `/api/v1/trend`
Category trend adjustment.

```bash
curl -X POST http://localhost:8000/api/v1/trend \
  -H "Content-Type: application/json" \
  -d '{"category": "Health & Fitness", "base_prediction": 4.1}'
```

### GET `/api/v1/history?limit=20`
Recent prediction history.

### GET `/api/v1/meta`
Categories, content ratings, model metrics, and trend data for all 33 categories.

### GET `/api/v1/health`
Health check.

---

## 🧠 ML Model

### Algorithm
**XGBoost Regressor** — gradient boosted trees, optimal for tabular regression.

### Features (10)

| Feature | Type | Engineering |
|---|---|---|
| `category` | Categorical | LabelEncoder → `category_enc` |
| `size_mb` | Float | Direct |
| `installs` | Integer | log1p → `log_installs` |
| `price` | Float | Direct |
| `content_rating` | Categorical | LabelEncoder → `content_rating_enc` |
| `reviews` | Integer | log1p → `log_reviews` |
| `update_days` | Integer | Direct |
| `num_screenshots` | Integer | Direct |
| `has_ads` | Binary | Direct |
| `is_free` | Binary | Derived from price |

### Performance
| Metric | Value |
|---|---|
| MAE | ~0.22 |
| RMSE | ~0.29 |
| R² | ~0.30 |

> Note: R² ~0.30 is expected for rating prediction — user ratings are inherently noisy and subjective. The model provides directional guidance, not exact regression.

### Explainability
Uses **SHAP TreeExplainer** for exact Shapley values. Falls back to permutation importance if SHAP is unavailable.

---

## 📊 Competitor Gap Analyzer

Uses **cosine similarity** on `[log(size), log(installs), price, log(reviews)]` to find the most similar apps in the category. Then computes:

- **Feature gaps**: Size, pricing, install count, review volume
- **Performance gaps**: Rating efficiency (stars per install unit)
- **Actionable insights**: Specific recommendations per gap
- **Profile radar**: Multi-dimension comparison chart

---

## 💬 AI Advisor

Intent detection covers 13 categories:
- `low_rating`, `improve_rating`, `improve_installs`
- `size_question`, `price_question`, `ads_question`
- `update_question`, `reviews_question`, `competitor_question`
- `category_question`, `screenshots`, `general_advice`

Response includes: issue detection with severity (critical/warning/info), root cause analysis, fix recommendations, and follow-up questions.

---

## 🌍 Trend Engine

Covers **33 app categories** with:
- `trend_score` (-1.0 to +1.0)
- `saturation` (0.0 to 1.0)
- `competition_level` (Low / Medium / High / Very High)
- `market_stage` (Emerging / Growing / Mature / Saturated / Declining / Niche)
- `yoy_growth` percentage

Adjustment formula:
```
total_adjustment = (trend_score × 0.12) + (-saturation × 0.08) + competition_penalty
adjusted_rating  = clamp(base_prediction + total_adjustment, 1.0, 5.0)
```

---

## 🗄️ Database

Three SQLite tables via SQLAlchemy ORM:

| Table | Columns |
|---|---|
| `prediction_logs` | id, input_features (JSON), prediction, confidence, trend_adjusted, timestamp |
| `chat_logs` | id, query, response (JSON), detected_intents, app_context, timestamp |
| `competitor_analysis_logs` | id, app_data (JSON), analysis_result (JSON), category, timestamp |

---

## 🎨 UI Themes

| Token | Dark | Light |
|---|---|---|
| Background | `#0A0F1E` | `#F8FAFC` |
| Card | `#1E293B` | `#FFFFFF` |
| Primary | `#6366F1` | `#4F46E5` |
| Accent | `#22C55E` | `#16A34A` |
| Danger | `#F87171` | `#DC2626` |

Font: **Inter** (300–800 weights)

---

## 🔮 Future Roadmap

- [ ] Real Play Store scraper (Playwright/SerpAPI)
- [ ] User authentication + personal prediction history
- [ ] Batch prediction via CSV upload
- [ ] Time-series rating trend prediction
- [ ] MLflow model experiment tracking
- [ ] LLM integration (OpenAI/Claude) for richer advisor responses
- [ ] App store review sentiment analysis (NLP)
- [ ] A/B test recommendation engine

---

*Built with FastAPI, Streamlit, XGBoost, SHAP, and Plotly*
>>>>>>> a50b2183 (RateIQ update)
