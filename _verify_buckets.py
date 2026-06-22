"""
Investigate classifier bucket mismatch:
clf_model produces 3 classes but code expects 5 (buckets 0-4).
"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from backend.services.model_service import ModelService
import numpy as np
import joblib

svc = ModelService("backend/models/model_artifacts.pkl")

print("=== CLASSIFIER CLASSES ===")
clf = svc.clf_model
print("clf classes_:       ", clf.classes_)
print("clf n_classes_:     ", clf.n_classes_)
print("Expected 5 classes: 0,1,2,3,4 (one per star bucket)")
print()
print("DIAGNOSIS: Classifier was trained with only %d unique buckets in training data" % len(clf.classes_))
print("This means some rating buckets had zero samples during training")
print()

# Check the actual rating distribution in the saved dataset
import os, pandas as pd

data_path = "data/apps.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    # normalise
    for col in ["Rating","rating"]:
        if col in df.columns:
            df["rating_num"] = pd.to_numeric(df[col], errors="coerce")
            break
    df = df.dropna(subset=["rating_num"])
    df["rating_num"] = df["rating_num"].clip(1.0, 5.0)
    
    # apply bucket function
    def bucket(r):
        return max(0, min(4, int(round(r)) - 1))
    df["bucket"] = df["rating_num"].apply(bucket)
    
    print("=== TRAINING DATA BUCKET DISTRIBUTION ===")
    dist = df["bucket"].value_counts().sort_index()
    for b, count in dist.items():
        stars = b + 1
        bar = "#" * (count // 50)
        print("  Bucket %d (~%d star): %5d rows  %s" % (b, stars, count, bar))
    
    print()
    print("=== RATING RANGE ===")
    print("  Min: %.1f  Max: %.1f  Mean: %.2f" % (df["rating_num"].min(), df["rating_num"].max(), df["rating_num"].mean()))
    
    missing_buckets = [b for b in [0,1,2,3,4] if b not in dist.index]
    print()
    if missing_buckets:
        print("MISSING BUCKETS (no training samples): %s" % missing_buckets)
        print("  -> These star levels have NO representative apps in training data")
        print("  -> LightGBMClassifier only learned %d classes instead of 5" % len(dist))
    else:
        print("All 5 buckets present in training data")

print()
print("=== IMPACT ON CONFIDENCE ===")
print("When bucket=4 (5-star apps), proba array has size=%d -> IndexError -> except block -> fallback to _confidence_boundary" % len(clf.classes_))
print("When bucket=3 (4-star apps like 4.48), it maps to class index that may not exist")
print()
print("Actual class mapping in clf:")
for i, c in enumerate(clf.classes_):
    print("  proba[%d] = class %d (~%d star apps)" % (i, c, c+1))
print()
print("=== PREDICTION BUCKET vs CLF CLASS MAP ===")
for rating in [1.0, 2.0, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0]:
    b = max(0, min(4, int(round(rating)) - 1))
    in_clf = b in clf.classes_
    print("  rating=%.1f -> bucket=%d -> in clf.classes_=%s %s"
          % (rating, b, in_clf, "" if in_clf else "<-- INDEX ERROR"))
