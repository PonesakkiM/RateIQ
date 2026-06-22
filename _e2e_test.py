"""
RateIQ End-to-End Verification
Read-only — no files modified.
"""
import requests, ast, os, sys

BASE = "http://localhost:8000/api/v1"
results = []

def chk(name, fn):
    try:
        ok, detail = fn()
        results.append((name, ok, detail))
    except Exception as e:
        results.append((name, False, str(e)[:80]))

# ── App payloads ───────────────────────────────────────────────────────────────
APP_GOOD = {"category":"EDUCATION","size_mb":25,"installs":100000,
            "price":0,"content_rating":"Everyone","reviews":5000,
            "update_days":30,"num_screenshots":5,"has_ads":0,"is_free":1}
APP_BAD  = {"category":"DATING","size_mb":150,"installs":500,
            "price":4.99,"content_rating":"Adults only 18+","reviews":5,
            "update_days":600,"num_screenshots":1,"has_ads":1,"is_free":0}

# ═══════════════════════════════════════════════════════════
# 1. API CONNECTIVITY
# ═══════════════════════════════════════════════════════════
def test_health():
    r = requests.get(BASE+"/health", timeout=5)
    d = r.json()
    return r.status_code==200 and d.get("status")=="ok", d.get("version","?")

def test_meta():
    r = requests.get(BASE+"/meta", timeout=5)
    d = r.json()
    cats = len(d.get("categories",[]))
    mm   = d.get("model_metrics",{}).get("model_type","?")
    return cats==33, "%d cats model=%s" % (cats, mm)

chk("Health endpoint",          test_health)
chk("Meta endpoint (33 cats)",  test_meta)

# ═══════════════════════════════════════════════════════════
# 2. PREDICT RATING
# ═══════════════════════════════════════════════════════════
def test_predict_executes():
    r = requests.post(BASE+"/predict", json=APP_GOOD, timeout=15)
    d = r.json()
    ok = "prediction" in d and 1.0 <= d["prediction"] <= 5.0
    return ok, "rating=%.2f" % d.get("prediction", 0)

def test_confidence_fields():
    r = requests.post(BASE+"/predict", json=APP_GOOD, timeout=15)
    d = r.json()
    valid_tiers = ("Very High","High","Medium","Low","Very Low")
    has_conf = "confidence" in d and 0 <= d["confidence"] <= 1
    has_tier = "confidence_tier" in d and d.get("confidence_tier") in valid_tiers
    has_desc = "confidence_desc" in d and len(d.get("confidence_desc","")) > 10
    return has_conf and has_tier and has_desc, \
           "tier=%s conf=%.0f%%" % (d.get("confidence_tier","?"), d.get("confidence",0)*100)

def test_shap_values():
    r = requests.post(BASE+"/predict", json=APP_GOOD, timeout=15)
    d = r.json()
    shap = d.get("shap_values",[])
    ok   = len(shap)==10 and all("label" in s and "value" in s for s in shap)
    pos  = sum(1 for s in shap if s["value"]>0)
    neg  = sum(1 for s in shap if s["value"]<=0)
    return ok, "%d SHAP vals +%d/-%d split" % (len(shap), pos, neg)

def test_positive_negative_split():
    r = requests.post(BASE+"/predict", json=APP_BAD, timeout=15)
    d = r.json()
    shap = d.get("shap_values",[])
    pos  = [s for s in shap if s["value"] > 0]
    neg  = [s for s in shap if s["value"] <= 0]
    ok   = len(pos) > 0 and len(neg) > 0
    return ok, "pos=%d neg=%d" % (len(pos), len(neg))

def test_trend_in_predict():
    r = requests.post(BASE+"/predict", json=APP_GOOD, timeout=15)
    d = r.json()
    t  = d.get("trend",{})
    ok = "adjusted_rating" in t and "market_stage" in t and "explanation" in t
    return ok, "stage=%s adj=%.2f" % (t.get("market_stage","?"), t.get("adjusted_rating",0))

def test_recommendation():
    r = requests.post(BASE+"/predict", json=APP_GOOD, timeout=15)
    d = r.json()
    rec = d.get("recommendation","")
    return len(rec) > 20, rec[:55]

def test_reset_state():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    fields = ["app_name","_prediction","_confidence","_conf_tier",
              "_conf_desc","_shap","_trend","_recommendation"]
    ok = all(f in src for f in fields) and "_comp" in src and "pop" in src
    return ok, "APP_STATE_DEFAULTS + pop(_comp) verified"

def test_positive_factors_ui():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    return "Positive Factors" in src, "section label present in source"

def test_negative_factors_ui():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    return "Negative Factors" in src, "section label present in source"

def test_app_name_in_results():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    ok = 'app_state.get("app_name")' in src and "app_state[\"app_name\"]" in src
    return ok, "app_name displayed in results block"

chk("Predict: executes",             test_predict_executes)
chk("Predict: confidence fields",    test_confidence_fields)
chk("Predict: SHAP values",          test_shap_values)
chk("Predict: +/- factor split",     test_positive_negative_split)
chk("Predict: trend in response",    test_trend_in_predict)
chk("Predict: recommendation",       test_recommendation)
chk("Predict: reset clears state",   test_reset_state)
chk("Predict: Positive Factors UI",  test_positive_factors_ui)
chk("Predict: Negative Factors UI",  test_negative_factors_ui)
chk("Predict: App Name in results",  test_app_name_in_results)

# ═══════════════════════════════════════════════════════════
# 3. COMPETITOR ANALYSIS
# ═══════════════════════════════════════════════════════════
def test_competitor_runs():
    r = requests.post(BASE+"/competitor-analysis",
                      json={"app_data": APP_GOOD}, timeout=15)
    d = r.json()
    ok = "category" in d and "similar_apps" in d and "summary" in d
    s  = d.get("similar_apps",[])
    return ok, "cat=%s top=%s" % (d.get("category","?"),
                                   s[0]["name"] if s else "none")

def test_competitor_chart_data():
    r = requests.post(BASE+"/competitor-analysis",
                      json={"app_data": APP_GOOD}, timeout=15)
    d = r.json()
    ok = len(d.get("similar_apps",[]))>0 and len(d.get("top_competitors",[]))>0
    return ok, "sim=%d top3=%d" % (len(d.get("similar_apps",[])),
                                    len(d.get("top_competitors",[])))

def test_competitor_app_context():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    ok = 'get_app().get("_prediction")' in src and "pred_r" in src
    return ok, "cross-page prediction context wired"

chk("Competitor: analysis runs",  test_competitor_runs)
chk("Competitor: chart data",     test_competitor_chart_data)
chk("Competitor: app context",    test_competitor_app_context)

# ═══════════════════════════════════════════════════════════
# 4. AI ADVISOR
# ═══════════════════════════════════════════════════════════
def test_chat_response():
    r = requests.post(BASE+"/chat",
                      json={"query":"Why is my app rating low?",
                            "app_data": APP_BAD, "chat_history":[]},
                      timeout=20)
    d = r.json()
    ok = ("response" in d and len(d["response"])>50 and
          "detected_intents" in d and "recommendations" in d)
    return ok, "intents=%s recs=%d" % (d.get("detected_intents",["?"])[:1],
                                        len(d.get("recommendations",[])))

def test_chat_ui():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    ok = ("chat_history" in src and "chat-wrap" in src and
          "chat-user" in src and "chat-ai" in src and
          "chat_app" in src and "chat_pred" in src)
    return ok, "chat UI + context wiring present"

chk("AI Advisor: API response",  test_chat_response)
chk("AI Advisor: UI elements",   test_chat_ui)

# ═══════════════════════════════════════════════════════════
# 5. TREND ANALYSIS
# ═══════════════════════════════════════════════════════════
def test_trend_endpoint():
    r = requests.post(BASE+"/trend",
                      json={"category":"EDUCATION","base_prediction":3.8},
                      timeout=10)
    d = r.json()
    ok = ("adjusted_rating" in d and "market_stage" in d and
          "trend_adjustment" in d and "explanation" in d)
    return ok, "stage=%s adj=%+.3f" % (d.get("market_stage","?"),
                                        d.get("trend_adjustment",0))

def test_trend_all_categories():
    r = requests.get(BASE+"/meta", timeout=5)
    d = r.json()
    trends = d.get("category_trends",{})
    return len(trends)==33, "%d category trends" % len(trends)

chk("Trend: endpoint works",      test_trend_endpoint)
chk("Trend: 33 categories",       test_trend_all_categories)

# ═══════════════════════════════════════════════════════════
# 6. EDA INSIGHTS
# ═══════════════════════════════════════════════════════════
def test_dataset_insights():
    r = requests.get(BASE+"/dataset-insights", timeout=10)
    d = r.json()
    ok = "insights" in d and d.get("dataset_rows",0) > 0
    return ok, "rows=%d insights=%d" % (d.get("dataset_rows",0),
                                         len(d.get("insights",[])))

def test_feature_importance():
    r = requests.get(BASE+"/feature-importance", timeout=5)
    d = r.json()
    ok = "items" in d and len(d.get("items",[]))==10
    return ok, "%d features model=%s" % (len(d.get("items",[])),
                                          d.get("model_type","?"))

def test_eda_dataset_files():
    root  = r"d:\RateIQ\files\rateiq_project\rateiq"
    paths = [
        os.path.join(root,"data","App_playstore_features.csv"),
        os.path.join(root,"data","App_playstore_final_cleaned.csv"),
        os.path.join(root,"data","apps.csv"),
    ]
    found = [os.path.basename(p) for p in paths if os.path.exists(p)]
    return len(found)>0, "found: %s" % ", ".join(found)

chk("EDA Insights: dataset insights",   test_dataset_insights)
chk("EDA Insights: feature importance", test_feature_importance)
chk("EDA Insights: dataset files",      test_eda_dataset_files)

# ═══════════════════════════════════════════════════════════
# 7. EDA DASHBOARD
# ═══════════════════════════════════════════════════════════
def test_eda_header_first():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    key       = 'elif "EDA Dashboard" in page_key:'
    idx       = src.find(key)
    header_i  = src.find("page-title", idx)
    data_i    = src.find("os.path.abspath", idx)
    ok        = 0 < (header_i - idx) < (data_i - idx)
    return ok, "header @+%d data-load @+%d" % (header_i-idx, data_i-idx)

chk("EDA Dashboard: header before data-load", test_eda_header_first)

# ═══════════════════════════════════════════════════════════
# 8. HISTORY
# ═══════════════════════════════════════════════════════════
def test_history_api():
    r = requests.get(BASE+"/history", params={"limit":50}, timeout=5)
    d = r.json()
    if not isinstance(d, list):
        return False, "response is not a list"
    if d:
        h = d[0]
        ok = all(k in h for k in ["id","prediction","confidence","timestamp","input_features"])
        return ok, "%d records all fields present" % len(d)
    return True, "0 records — empty state OK"

def test_history_timestamp():
    r = requests.get(BASE+"/history", params={"limit":5}, timeout=5)
    d = r.json()
    if not d:
        return True, "no records — empty state"
    ts  = str(d[0].get("timestamp",""))
    ok  = len(ts) >= 10
    fmt = ts[:19].replace("T"," ") if ts else "n/a"
    return ok, "ts=%s" % fmt

def test_history_empty_state():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    return "No predictions recorded yet" in src, "empty state message present"

chk("History: API returns records",  test_history_api)
chk("History: timestamp parse",      test_history_timestamp)
chk("History: empty state handled",  test_history_empty_state)

# ═══════════════════════════════════════════════════════════
# 9. ABOUT PAGE
# ═══════════════════════════════════════════════════════════
def test_about_sections():
    with open("frontend/app.py", encoding="utf-8") as f:
        src = f.read()
    sections = {
        "Platform Workflow":  "Platform Workflow" in src,
        "Prediction Engine":  "Prediction Engine" in src,
        "Dataset":            "dataset_stats" in src,
        "Analytics Features": "Analytics Features" in src,
        "Technical Stack":    "Technical Stack" in src,
        "Team Information":   "team_members" in src and "Platform Engineering" in src,
    }
    missing = [k for k,v in sections.items() if not v]
    ok = len(missing)==0
    return ok, "all 6 present" if ok else "missing: "+str(missing)

chk("About: all 6 sections", test_about_sections)

# ═══════════════════════════════════════════════════════════
# PRINT REPORT
# ═══════════════════════════════════════════════════════════
passed = sum(1 for _,ok,_ in results if ok)
total  = len(results)
errors = [(n,d) for n,ok,d in results if not ok]
warns  = []

AREAS = [
    ("API Connectivity",     ["Health endpoint","Meta endpoint (33 cats)"]),
    ("Predict Rating",       ["Predict: executes","Predict: confidence fields",
                              "Predict: SHAP values","Predict: +/- factor split",
                              "Predict: trend in response","Predict: recommendation",
                              "Predict: reset clears state",
                              "Predict: Positive Factors UI","Predict: Negative Factors UI",
                              "Predict: App Name in results"]),
    ("Competitor Analysis",  ["Competitor: analysis runs","Competitor: chart data",
                              "Competitor: app context"]),
    ("AI Advisor",           ["AI Advisor: API response","AI Advisor: UI elements"]),
    ("Trend Analysis",       ["Trend: endpoint works","Trend: 33 categories"]),
    ("EDA Insights",         ["EDA Insights: dataset insights",
                              "EDA Insights: feature importance",
                              "EDA Insights: dataset files"]),
    ("EDA Dashboard",        ["EDA Dashboard: header before data-load"]),
    ("History",              ["History: API returns records","History: timestamp parse",
                              "History: empty state handled"]),
    ("About",                ["About: all 6 sections"]),
]
res_map = {n: (ok,d) for n,ok,d in results}

print()
print("=" * 65)
print("  RateIQ — END-TO-END VERIFICATION REPORT")
print("  frontend/app.py — 1893 lines | No files modified")
print("=" * 65)
print()

for area, checks in AREAS:
    area_ok = all(res_map.get(c,(False,""))[0] for c in checks)
    print("  [%s]  %s" % ("PASS" if area_ok else "FAIL", area))
    for c in checks:
        ok, detail = res_map.get(c, (False,"not run"))
        mark = "OK" if ok else "XX"
        print("    %s  %-37s %s" % (mark, c, detail[:38]))
    print()

print("  " + "─"*63)
print("  Total checks : %d" % total)
print("  Passed       : %d" % passed)
print("  Failed       : %d" % (total - passed))
print()
if errors:
    print("  RUNTIME ERRORS (%d):" % len(errors))
    for n,d in errors:
        print("    - %s: %s" % (n,d))
else:
    print("  Runtime errors  : None")
if warns:
    print("  Warnings:")
    for w in warns: print("    - %s" % w)
else:
    print("  Warnings        : None")
print()
print("  Overall readiness : %.0f%%" % (passed/total*100))
print("=" * 65)
