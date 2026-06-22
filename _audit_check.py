import requests, time, os, sqlite3

base = "http://localhost:8000/api/v1"

print("=== CONFIDENCE TEST ===")
apps = [
    ("Worst-case",  {"category":"DATING","size_mb":80,"installs":1000,"price":2.99,"content_rating":"Mature 17+","reviews":10,"update_days":500,"num_screenshots":1,"has_ads":1,"is_free":0}),
    ("Average",     {"category":"TOOLS","size_mb":25,"installs":100000,"price":0,"content_rating":"Everyone","reviews":5000,"update_days":60,"num_screenshots":4,"has_ads":0,"is_free":1}),
    ("Best-case",   {"category":"EDUCATION","size_mb":15,"installs":1000000,"price":0,"content_rating":"Everyone","reviews":50000,"update_days":14,"num_screenshots":7,"has_ads":0,"is_free":1}),
]
for name, app in apps:
    r = requests.post(base+"/predict", json=app, timeout=10).json()
    print("  %-12s: %.2f stars | conf=%.3f | tier=%s" % (name, r["prediction"], r["confidence"], r["confidence_tier"]))

print()
print("=== HISTORY CHECK ===")
hist = requests.get(base+"/history?limit=20", timeout=5).json()
print("  Records in DB: %d" % len(hist))
if hist:
    h = hist[0]
    print("  Latest ID=%d  prediction=%.2f  ts=%s" % (h["id"], h["prediction"], str(h["timestamp"])[:19]))

print()
print("=== DOUBLE-INFERENCE TIMING ===")
app = {"category":"TOOLS","size_mb":25,"installs":100000,"price":0,"content_rating":"Everyone","reviews":5000,"update_days":30,"num_screenshots":5,"has_ads":0,"is_free":1}
t0 = time.time()
requests.post(base+"/predict", json=app, timeout=10)
elapsed = (time.time() - t0) * 1000
print("  /predict roundtrip: %.0f ms" % elapsed)

print()
print("=== DB FILE CHECK ===")
db_path = os.path.join("rateiq.db")
if os.path.exists(db_path):
    size = os.path.getsize(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_logs")
    count = cur.fetchone()[0]
    conn.close()
    print("  DB size: %d bytes | prediction_logs rows: %d" % (size, count))
else:
    print("  rateiq.db NOT FOUND at project root")
    # Check other locations
    for p in ["rateiq.db", "backend/rateiq.db"]:
        if os.path.exists(p):
            print("  Found at:", p, "size:", os.path.getsize(p))

print()
print("=== STALE ARTIFACT CHECK ===")
for p in ["model.joblib", "backend/models/model_artifacts.pkl", "backend/models/model_artifacts.joblib"]:
    if os.path.exists(p):
        print("  %s: %.1f KB" % (p, os.path.getsize(p)/1024))
    else:
        print("  %s: NOT FOUND" % p)
