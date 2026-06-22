"""
RateIQ – Model Artifact Verification Script
Checks: which file is loaded, load order, model type, artifact integrity
"""
import sys, os, time, hashlib, pickle
sys.path.insert(0, ".")

print("=" * 65)
print("  RateIQ – Model Artifact Verification")
print("=" * 65)

# ── 1. Physical file inventory ─────────────────────────────────────────────
print("\n[1] PHYSICAL FILE INVENTORY")
candidates = {
    "ROOT stale":              "model.joblib",
    "BACKEND pkl (primary)":   "backend/models/model_artifacts.pkl",
    "BACKEND joblib (primary)":"backend/models/model_artifacts.joblib",
}
file_info = {}
for label, path in candidates.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        import datetime
        mdt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        # MD5 for identity check
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()[:12]
        file_info[path] = {"size": size, "mtime": mdt, "md5": md5}
        print("  %-32s  %s  %8.1f KB  md5=%s  modified=%s"
              % (label, path, size/1024, md5, mdt))
    else:
        print("  %-32s  NOT FOUND" % label)

# ── 2. Simulate ModelService._load_artifacts() load order ─────────────────
print("\n[2] LOAD ORDER SIMULATION (mirrors ModelService._load_artifacts)")
from pathlib import Path

model_path = Path("backend/models/model_artifacts.pkl").resolve()
print("  Config model_path resolved to: %s" % model_path)
print()

load_attempts = [
    model_path,
    model_path.with_suffix(".joblib"),
    model_path.parent / "model_artifacts.joblib",
]
print("  Load attempt sequence:")
winner = None
for i, candidate in enumerate(load_attempts, 1):
    exists = candidate.exists()
    marker = ">>> WOULD LOAD <<<" if (exists and winner is None) else ("EXISTS but skipped" if exists else "not found")
    if exists and winner is None:
        winner = candidate
    print("  %d. %-55s  %s" % (i, str(candidate), marker))

print()
print("  ACTUAL FILE THAT WILL BE LOADED: %s" % winner)

# ── 3. Load the winning artifact and inspect ───────────────────────────────
print("\n[3] LOADING WINNER ARTIFACT")
if winner is None:
    print("  ERROR: No artifact found — cannot continue")
    sys.exit(1)

t0 = time.time()
try:
    import joblib
    arts = joblib.load(winner)
    load_method = "joblib"
    load_time = (time.time() - t0) * 1000
    print("  Loaded via joblib in %.0f ms" % load_time)
except Exception as e1:
    print("  joblib failed: %s — trying pickle" % e1)
    with open(winner, "rb") as f:
        arts = pickle.load(f)
    load_method = "pickle"
    load_time = (time.time() - t0) * 1000
    print("  Loaded via pickle in %.0f ms" % load_time)

# ── 4. Artifact contents ────────────────────────────────────────────────────
print("\n[4] ARTIFACT CONTENTS")
for key in arts.keys():
    val = arts[key]
    if hasattr(val, "__class__"):
        type_name = type(val).__name__
        module = getattr(type(val), "__module__", "?")
    else:
        type_name = str(type(val))
        module = "?"
    extra = ""
    if key == "model":
        extra = "  ← PRIMARY REGRESSOR"
        try:
            extra += "  [%s from %s]" % (type_name, module)
        except Exception:
            pass
    elif key == "clf_model":
        if val is None:
            extra = "  ← CLASSIFIER = None (confidence will use boundary fallback)"
        else:
            extra = "  ← CLASSIFIER present ✅"
    elif key == "metrics":
        pass
    print("  %-25s  %-30s%s" % (key, type_name, extra))

# ── 5. Model type + metrics ────────────────────────────────────────────────
print("\n[5] MODEL TYPE & METRICS")
model = arts["model"]
clf   = arts.get("clf_model")
print("  Primary model type  : %s" % type(model).__name__)
print("  Primary model module: %s" % getattr(type(model), "__module__", "?"))
print("  Classifier present  : %s" % ("YES — %s" % type(clf).__name__ if clf is not None else "NO — None"))
print()
metrics = arts.get("metrics", {})
for k, v in metrics.items():
    if k != "confusion_matrix":
        print("  %-25s: %s" % (k, v))

# ── 6. Feature set ────────────────────────────────────────────────────────
print("\n[6] FEATURE CONFIGURATION")
print("  Features  : %s" % arts.get("features", []))
print("  Categories: %d  (sample: %s...)" % (
    len(arts.get("categories", [])),
    str(arts.get("categories", [])[:3])
))
print("  Content ratings: %s" % arts.get("content_ratings", []))

# ── 7. Verify the ROOT artifact is different ─────────────────────────────
print("\n[7] ROOT ARTIFACT IDENTITY CHECK")
root_path = "model.joblib"
if os.path.exists(root_path):
    try:
        import joblib as jl
        root_arts = jl.load(root_path)
        root_type = type(root_arts).__name__
        if isinstance(root_arts, dict):
            root_model_type = type(root_arts.get("model", "?")).__name__
            root_has_metrics = "metrics" in root_arts
        else:
            root_model_type = root_type
            root_has_metrics = False
        print("  Root model.joblib contains: %s" % root_type)
        if isinstance(root_arts, dict):
            print("  Root model type inside dict: %s" % root_model_type)
            print("  Root keys: %s" % list(root_arts.keys()))
        else:
            print("  Root object type: %s" % root_type)
            print("  Root module     : %s" % getattr(type(root_arts), "__module__", "?"))
        prod_path = "backend/models/model_artifacts.joblib"
        if os.path.exists(prod_path):
            with open(root_path, "rb") as f1, open(prod_path, "rb") as f2:
                same = hashlib.md5(f1.read()).hexdigest() == hashlib.md5(f2.read()).hexdigest()
            print("  IDENTICAL to production artifact: %s" % same)
    except Exception as e:
        print("  Failed to load root artifact: %s" % e)
else:
    print("  Root model.joblib: NOT FOUND")

# ── 8. Live prediction test ─────────────────────────────────────────────────
print("\n[8] LIVE PREDICTION SMOKE TEST")
try:
    from backend.services.model_service import ModelService
    svc = ModelService("backend/models/model_artifacts.pkl")
    test_data = {
        "category": "EDUCATION", "size_mb": 25.0, "installs": 100000,
        "price": 0.0, "content_rating": "Everyone", "reviews": 5000,
        "update_days": 30, "num_screenshots": 5, "has_ads": 0, "is_free": 1
    }
    t0 = time.time()
    pred, conf, shap = svc.predict(test_data)
    t1 = time.time()
    print("  Prediction  : %.2f stars" % pred)
    print("  Confidence  : %.3f (%.0f%%)" % (conf, conf*100))
    print("  Inference ms: %.0f" % ((t1-t0)*1000))
    print("  clf_model   : %s" % ("loaded" if svc.clf_model else "None — using boundary fallback"))
    print("  SHAP engine : %s" % ("TreeExplainer" if svc._shap_explainer else "Permutation fallback"))
    tier_info = svc._confidence_tier(conf)
    print("  Tier        : %s" % tier_info["tier"])

    # test boundary formula directly
    print()
    print("  _confidence_boundary values (shows inversion issue):")
    for r in [1.0, 2.0, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0]:
        c = svc._confidence_boundary(r)
        tier = svc._confidence_tier(c)["tier"]
        bar = "#" * int(c * 30)
        print("    rating=%.1f  conf=%.3f  tier=%-10s  %s" % (r, c, tier, bar))
except Exception as e:
    print("  Smoke test failed: %s" % e)

print()
print("=" * 65)
print("  Verification complete.")
print("=" * 65)
