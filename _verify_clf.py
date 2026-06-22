"""
Diagnose why clf_model produces 0.450 even though it IS loaded.
"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from backend.services.model_service import ModelService
import numpy as np

svc = ModelService("backend/models/model_artifacts.pkl")

# Build a test feature vector manually
test_data = {
    "category": "EDUCATION", "size_mb": 25.0, "installs": 100000,
    "price": 0.0, "content_rating": "Everyone", "reviews": 5000,
    "update_days": 30, "num_screenshots": 5, "has_ads": 0, "is_free": 1
}
vec = svc._build_feature_vector(test_data)
vec_scaled = svc.scaler.transform(vec.reshape(1, -1))[0]

raw_pred = float(svc.model.predict(vec_scaled.reshape(1, -1))[0])
print("Raw prediction: %.4f" % raw_pred)
print("Clipped:        %.2f" % np.clip(raw_pred, 1.0, 5.0))

print()
print("=== CLASSIFIER PROBE ===")
print("clf_model present:", svc.clf_model is not None)
if svc.clf_model is not None:
    try:
        proba = svc.clf_model.predict_proba(vec_scaled.reshape(1, -1))[0]
        print("proba shape:", proba.shape)
        print("proba values:", [round(p,4) for p in proba])
        print("proba sum:   ", sum(proba))
        
        pred_rating = round(float(np.clip(raw_pred, 1.0, 5.0)), 2)
        bucket = max(0, min(4, int(round(pred_rating)) - 1))
        print()
        print("pred_rating:", pred_rating)
        print("bucket:     ", bucket, "(0=1star,1=2star,2=3star,3=4star,4=5star)")
        print("p_main:     ", round(proba[bucket], 4))
        p_adj = (proba[bucket-1] if bucket > 0 else 0) + (proba[bucket+1] if bucket < 4 else 0)
        print("p_adj:      ", round(p_adj, 4))
        conf_raw = proba[bucket] + 0.4 * p_adj
        print("conf_raw:   ", round(conf_raw, 4))
        conf_clipped = float(np.clip(conf_raw, 0.40, 0.99))
        print("conf_clip:  ", round(conf_clipped, 3))
        
        # show all bucket proba
        print()
        print("Bucket distribution:")
        for i, p in enumerate(proba):
            star = i + 1
            bar = "#" * int(p * 60)
            marker = " <<< predicted" if i == bucket else ""
            print("  Bucket %d (~%d star): %.4f  %s%s" % (i, star, p, bar, marker))
    except Exception as e:
        print("clf predict_proba FAILED:", type(e).__name__, e)
        import traceback
        traceback.print_exc()

print()
print("=== TEST MULTIPLE RATINGS ===")
test_cases = [
    ("Very poor",   {"category":"DATING","size_mb":150,"installs":100,"price":4.99,"content_rating":"Adults only 18+","reviews":5,"update_days":700,"num_screenshots":1,"has_ads":1,"is_free":0}),
    ("Poor",        {"category":"DATING","size_mb":80,"installs":1000,"price":2.99,"content_rating":"Mature 17+","reviews":50,"update_days":400,"num_screenshots":1,"has_ads":1,"is_free":0}),
    ("Average",     {"category":"TOOLS","size_mb":30,"installs":50000,"price":0,"content_rating":"Everyone","reviews":1000,"update_days":90,"num_screenshots":3,"has_ads":0,"is_free":1}),
    ("Good",        {"category":"EDUCATION","size_mb":20,"installs":500000,"price":0,"content_rating":"Everyone","reviews":20000,"update_days":30,"num_screenshots":6,"has_ads":0,"is_free":1}),
    ("Excellent",   {"category":"EDUCATION","size_mb":12,"installs":5000000,"price":0,"content_rating":"Everyone","reviews":200000,"update_days":7,"num_screenshots":8,"has_ads":0,"is_free":1}),
]
for name, app in test_cases:
    v = svc._build_feature_vector(app)
    vs = svc.scaler.transform(v.reshape(1,-1))[0]
    raw = float(svc.model.predict(vs.reshape(1,-1))[0])
    pred = round(float(np.clip(raw,1,5)),2)
    
    # classifier path
    conf_clf = svc._confidence_from_clf(vs, pred)
    conf_bnd = svc._confidence_boundary(pred)
    tier = svc._confidence_tier(conf_clf)["tier"]
    
    # show proba if clf is available
    if svc.clf_model:
        try:
            proba = svc.clf_model.predict_proba(vs.reshape(1,-1))[0]
            bucket = max(0, min(4, int(round(pred)) - 1))
            p_main = proba[bucket]
        except Exception:
            p_main = -1.0
    else:
        p_main = -1.0
    
    print("  %-12s rating=%.2f  clf_conf=%.3f  bnd_conf=%.3f  p_bucket=%.4f  tier=%s" 
          % (name, pred, conf_clf, conf_bnd, p_main, tier))
