import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Dummy dataset (you can replace later with real data)
data = {
    "rating_count": [100, 200, 300, 400, 500],
    "reviews": [10, 20, 30, 40, 50],
    "size": [15, 20, 25, 30, 35],
    "installs": [1000, 2000, 3000, 4000, 5000],
    "rating": [3.5, 4.0, 4.2, 4.5, 4.8]
}

df = pd.DataFrame(data)

X = df.drop("rating", axis=1)
y = df["rating"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, "model.joblib")

print("✅ Model trained and saved as model.joblib")