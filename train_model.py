"""
Creates a reproducible demonstration model for the deployment practical.
The generated training data is synthetic and is only for learning/demo purposes.
Replace it with your faculty-provided dataset if one is required.
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

rng = np.random.default_rng(42)
n = 800
study_hours = rng.uniform(0, 10, n)
attendance = rng.uniform(30, 100, n)
previous_score = rng.uniform(0, 100, n)
assignment_score = rng.uniform(0, 100, n)

noise = rng.normal(0, 7, n)
score = (
    3.0 * study_hours
    + 0.25 * attendance
    + 0.30 * previous_score
    + 0.25 * assignment_score
    - 20
    + noise
)
score = np.clip(score, 0, 100)

X = np.column_stack([study_hours, attendance, previous_score, assignment_score])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, score)

out = Path(__file__).parent / "student_result_model.joblib"
joblib.dump(model, out)
print(f"Saved demo model to {out}")
