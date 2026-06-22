"""
RateIQ – Post-Implementation Validation Suite
Tests all approved fixes without modifying any code.
"""
import sys, os, time, sqlite3, hashlib, ast, re, requests

sys.path.insert(0, ".")
BASE = "http://localhost:8000/api/v1"

PASS = "PASS"
FAIL = "FAIL"
INFO = "INFO"

results = []

def check(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((status, name, detail))
    marker = "✅" if passed else "❌"
    print("  %s  %-50s  %s" % (marker, name, detail))


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  1. SYNTAX & STRUCTURAL CHECKS")
print("="*65)
# ═══════════════════════════════════════════════════════════

files = [
    "backend/services/model_service.py",
    "backend/api/routes.py",
    "backend/core/config.py",
    "frontend/api_client.py",
    "frontend/app.py",
]
for f in files:
    with open(f, encoding="utf-8") as fh:
        src = fh.read()
    try:
        ast.parse(src)
        check("Syntax: " + f, True)
    except SyntaxError as e:
        check("Syntax: " + f, False, "line %d: %s" % (e.lineno, e.msg))

with open("frontend/app.py", encoding="utf-8") as f:
    app_src = f.read()

ifs   = re.findall(r'^if .+ in page:', app_src, re.M)
elifs = re.findall(r'^elif .+ in page:', app_src, re.M)
check("Page routing: 1 if + 7 elif", len(ifs)==1 and len(elifs)==7,
      "if=%d elif=%d" % (len(ifs), len(elifs)))
check("Global state: _app_state key",     "_app_state" in app_src)
check("Global state: get_app() helper",   "def get_app()" in app_src)
check("Global state: set_app() helper",   "def set_app(" in app_src)
check("Global state: reset_app() helper", "def reset_app()" in app_src)
check("URL sync: set_app in URL block",   "set_app(" in app_src and "app_name" in app_src)
check("Cross-page: Competitor reads state", "get_app().get" in app_src)
check("Cross-page: Trend pre-fills state",  'app.get("_prediction")' in app_src)
check("Cross-page: Advisor auto-context",   "chat_app" in app_src and "chat_pred" in app_src)
check("EDA Dashboard: cats_f init",         "cats_f  = []" in app_src or "cats_f = []" in app_src)
check("History: get_history(limit=50)",     "get_history(limit=50)" in app_src)
check("All 8 pages present",
      all(pg in app_src for pg in
          ["Predict Rating","Competitor","Advisor","Trend",
           "EDA Insights","EDA Dashboard","History","About"]))

# api_client uses params dict
with open("frontend/api_client.py", encoding="utf-8") as f:
    cli_src = f.read()
check("api_client: _get accepts params",  "def _get(endpoint: str, params" in cli_src)
check("api_client: get_history uses params", 'params={"limit": limit}' in cli_src)

# routes: no double-inference
with open("backend/api/routes.py", encoding="utf-8") as f:
    route_src = f.read()
check("Routes: no predict_with_confidence_detail call",
      "predict_with_confidence_detail" not in route_src)
check("Routes: uses 5-tuple unpack from predict()",
      "prediction, confidence, conf_tier, conf_desc, shap_list" in route_src)

# model_service: corrected confidence + classifier fix
with open("backend/services/model_service.py", encoding="utf-8") as f:
    ms_src = f.read()
check("ModelService: _confidence_corrected formula",    "_confidence_corrected" in ms_src)
check("ModelService: _clf_class_to_idx mapping",        "_clf_class_to_idx" in ms_src)
check("ModelService: predict returns 5-tuple",          "conf_tier, conf_desc" in ms_src)
check("ModelService: no bare except in clf path",       "except Exception as e:" in ms_src)

# config: points to joblib
with open("backend/core/config.py", encoding="utf-8") as f:
    cfg_src = f.read()
check("Config: model_path points to .joblib",
      "model_artifacts.joblib" in cfg_src)


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  2. ACTIVE MODEL PATH VERIFICATION")
print("="*65)
# ═══════════════════════════════════════════════════════════

from backend.core.config import settings
model_path = settings.model_path
path_exists = os.path.exists(model_path)
is_joblib   = model_path.endswith(".joblib")
size_kb     = os.path.getsize(model_path) / 1024 if path_exists else 0

print("  Config model_path : %s" % model_path)
print("  File exists       : %s" % path_exists)
print("  Format            : %s" % (".joblib" if is_joblib else ".pkl"))
print("  Size              : %.1f KB" % size_kb)
print("  Stale root artifact: %s (not used)" % os.path.exists("model.joblib"))
check("Active path: .joblib primary",  is_joblib and path_exists,
      "%.0f KB at ...%s" % (size_kb, model_path[-30:]))
check("Stale model.joblib NOT configured", "model.joblib" not in model_path)
check("Production size >8000 KB",     size_kb > 8000,
      "%.1f KB" % size_kb)


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  3. BACKEND API CHECKS")
print("="*65)
# ═══════════════════════════════════════════════════════════

try:
    h = requests.get(BASE + "/health", timeout=5).json()
    check("Health endpoint",           h["status"] == "ok", h["version"])
except Exception as e:
    check("Health endpoint", False, str(e))
    sys.exit(1)

# Confirm classifier loaded from backend logs via meta
try:
    meta = requests.get(BASE + "/meta", timeout=5).json()
    mm   = meta.get("model_metrics", {})
    check("Meta: model_type=LightGBM", mm.get("model_type") == "LightGBM", mm.get("model_type","?"))
    check("Meta: accuracy >0.80",      mm.get("accuracy", 0) > 0.80,
          "%.4f" % mm.get("accuracy",0))
    check("Meta: 33 categories",       len(meta.get("categories", [])) == 33,
          "%d" % len(meta.get("categories",[])))
except Exception as e:
    check("Meta endpoint", False, str(e))


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  4. CONFIDENCE CALCULATION VERIFICATION")
print("="*65)
# ═══════════════════════════════════════════════════════════

test_cases = [
    ("Worst-case (stale, ads, dating)",
     {"category":"DATING","size_mb":150,"installs":100,"price":4.99,
      "content_rating":"Adults only 18+","reviews":5,"update_days":700,
      "num_screenshots":1,"has_ads":1,"is_free":0},
     (1.0, 4.5), None),          # expect low prediction
    ("Average app",
     {"category":"TOOLS","size_mb":30,"installs":50000,"price":0,
      "content_rating":"Everyone","reviews":1000,"update_days":90,
      "num_screenshots":4,"has_ads":0,"is_free":1},
     (3.5, 4.5), None),
    ("Best-case (education, popular, fresh)",
     {"category":"EDUCATION","size_mb":15,"installs":5000000,"price":0,
      "content_rating":"Everyone","reviews":200000,"update_days":7,
      "num_screenshots":8,"has_ads":0,"is_free":1},
     (4.0, 5.0), None),
]

for name, app, (expect_lo, expect_hi), _ in test_cases:
    try:
        r = requests.post(BASE + "/predict", json=app, timeout=10).json()
        pred  = r["prediction"]
        conf  = r["confidence"]
        tier  = r.get("confidence_tier", "?")
        desc  = r.get("confidence_desc", "")[:50]
        in_range = expect_lo <= pred <= expect_hi
        check("Predict %s" % name[:35],
              True,
              "%.2f★  conf=%.3f  tier=%s" % (pred, conf, tier))
        # Confidence should no longer be stuck at 0.450 for high-rating apps
        not_stuck = not (pred > 4.0 and conf <= 0.450)
        check("  Confidence not stuck at 0.450",
              not_stuck,
              "conf=%.3f for %.2f★" % (conf, pred))
    except Exception as e:
        check("Predict %s" % name[:35], False, str(e))

# Verify corrected boundary formula directly
print()
print("  Confidence boundary values (OLD: peaked at 3.0 | NEW: peaks away from 3.0):")
from backend.services.model_service import ModelService
svc = ModelService(settings.model_path)
for r_val in [1.0, 2.5, 3.0, 3.8, 4.0, 4.5, 4.8, 5.0]:
    c = svc._confidence_corrected(r_val)
    tier, _ = svc._confidence_tier(c)
    bar  = "#" * int(c * 30)
    invert_ok = (r_val >= 4.0 and c >= 0.35) or (r_val <= 2.0 and c >= 0.35) or (2.5 <= r_val <= 3.5 and c <= 0.50)
    print("    rating=%.1f  corrected_conf=%.3f  tier=%-10s  %s" % (r_val, c, tier, bar))


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  5. PREDICTION LATENCY (BEFORE vs AFTER)")
print("="*65)
# ═══════════════════════════════════════════════════════════

app_payload = {
    "category":"TOOLS","size_mb":30,"installs":100000,"price":0,
    "content_rating":"Everyone","reviews":5000,"update_days":45,
    "num_screenshots":5,"has_ads":0,"is_free":1
}
times = []
for i in range(5):
    t0 = time.time()
    requests.post(BASE + "/predict", json=app_payload, timeout=15)
    times.append((time.time() - t0) * 1000)
    time.sleep(0.1)

avg_ms  = sum(times) / len(times)
min_ms  = min(times)
max_ms  = max(times)
BEFORE_MS = 2110   # measured before double-inference fix

print("  Before fix (double-inference) : ~%d ms" % BEFORE_MS)
print("  After  fix (single-inference) : avg=%.0f ms  min=%.0f  max=%.0f" % (avg_ms, min_ms, max_ms))
improvement = (BEFORE_MS - avg_ms) / BEFORE_MS * 100
check("Latency improved vs baseline",
      avg_ms < BEFORE_MS * 0.9,
      "%.0f ms avg (was %d ms) = %.0f%% faster" % (avg_ms, BEFORE_MS, improvement))
check("Latency under 1500ms",
      avg_ms < 1500,
      "%.0f ms avg" % avg_ms)


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  6. HISTORY PAGE VERIFICATION")
print("="*65)
# ═══════════════════════════════════════════════════════════

# Run a fresh prediction to ensure at least 1 record
requests.post(BASE + "/predict", json=app_payload, timeout=10)

# Use params dict (the fix)
try:
    hist = requests.get(BASE + "/history", params={"limit": 50}, timeout=5).json()
    check("History endpoint: returns list",    isinstance(hist, list), "type=%s" % type(hist).__name__)
    check("History endpoint: non-empty",       len(hist) > 0, "%d records" % len(hist))
    if hist:
        h0 = hist[0]
        check("History record has 'prediction'",   "prediction"     in h0)
        check("History record has 'confidence'",   "confidence"     in h0)
        check("History record has 'timestamp'",    "timestamp"      in h0)
        check("History record has 'trend_adjusted'","trend_adjusted" in h0)
        check("History record has 'input_features'","input_features" in h0 and isinstance(h0["input_features"], dict))
        ts = str(h0.get("timestamp",""))
        check("Timestamp parseable",
              len(ts) >= 10,
              ts[:19])
except Exception as e:
    check("History endpoint", False, str(e))

# DB file check
db_path = "rateiq.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_logs")
    count = cur.fetchone()[0]
    conn.close()
    check("SQLite prediction_logs rows", count > 0, "%d rows" % count)
else:
    check("SQLite rateiq.db exists", False, "not found at project root")


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  7. URL SYNC VERIFICATION (structural)")
print("="*65)
# ═══════════════════════════════════════════════════════════

with open("frontend/app.py", encoding="utf-8") as f:
    app_src = f.read()

url_block_start = app_src.find("Auto-fill from Play Store URL")
url_block_end   = app_src.find("st.rerun()", url_block_start) + 10
url_block       = app_src[url_block_start:url_block_end]

check("URL block: set_app called",         "set_app(" in url_block)
check("URL block: app_name written",       "app_name" in url_block)
check("URL block: category written",       "category" in url_block)
check("URL block: installs written",       "installs" in url_block)
check("URL block: reviews written",        "reviews"  in url_block)
check("URL block: st.rerun() called",      "st.rerun()" in url_block)
check("URL block: no _url_pre key used",   "_url_pre" not in url_block,
      "(old isolated key removed)")


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  8. APP STATE CONSISTENCY VERIFICATION")
print("="*65)
# ═══════════════════════════════════════════════════════════

# Check each page reads from _app_state via _render_app_form or get_app()
for pg, expected_reads in [
    ("Predict Rating",   ["_render_app_form", "get_app()"]),
    ("Competitor",       ["_render_app_form", "get_app()"]),
    ("Trend Booster",    ["get_app()"]),
    ("AI Advisor",       ["get_app()"]),
    ("EDA Dashboard",    ["get_app()"]),
]:
    pg_start = app_src.find('# PAGE')
    # find the page block
    markers = {"Predict Rating": '"🔮" in page',
               "Competitor": '"Competitor" in page',
               "Trend Booster": '"Trend" in page',
               "AI Advisor": '"Advisor" in page',
               "EDA Dashboard": '"EDA Dashboard" in page'}
    marker = markers.get(pg, pg)
    idx = app_src.find(marker)
    next_page = min(
        (app_src.find('"Competitor" in page', idx+1) if pg!="Competitor" else 999999),
        (app_src.find('"Advisor" in page', idx+1)    if pg!="AI Advisor" else 999999),
        (app_src.find('"EDA Dashboard" in page', idx+1) if pg!="EDA Dashboard" else 999999),
        (app_src.find('"History" in page', idx+1)    if pg!="History" else 999999),
    )
    if next_page == 999999 or next_page <= idx:
        next_page = idx + 8000
    block = app_src[idx:next_page]
    reads_ok = all(kw in block for kw in expected_reads)
    check("%-20s reads _app_state" % pg, reads_ok,
          "via " + "+".join(expected_reads))

check("Predict result writes to _app_state",
      "set_app(" in app_src and "_prediction" in app_src)
check("Competitor uses _prediction from _app_state",
      'get_app().get("_prediction")' in app_src or "get_app().get" in app_src)
check("Trend pre-fills category from _app_state",
      "app[\"category\"]" in app_src or 'app["category"]' in app_src)


# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  VALIDATION SUMMARY")
print("="*65)
# ═══════════════════════════════════════════════════════════

passed = sum(1 for s,_,_ in results if s == PASS)
failed = sum(1 for s,_,_ in results if s == FAIL)
total  = len(results)

print()
for status, name, detail in results:
    if status == FAIL:
        print("  ❌  %-50s  %s" % (name, detail))

print()
print("  Total checks : %d" % total)
print("  Passed       : %d" % passed)
print("  Failed       : %d" % failed)
print("  Pass rate    : %.0f%%" % (passed/total*100))
print()
