from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).parent / "student_result_model.joblib"

app = FastAPI(
    title="Student Result Prediction API",
    description="Demo API for predicting a student's result from study and assessment features.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=24, description="Average study hours per day")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage")
    previous_score: float = Field(..., ge=0, le=100, description="Previous exam score")
    assignment_score: float = Field(..., ge=0, le=100, description="Assignment score")

@app.get("/")
def root():
    return {
        "message": "Student Result Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(student: StudentInput):
    features = np.array([[
        student.study_hours,
        student.attendance,
        student.previous_score,
        student.assignment_score,
    ]])
    predicted_score = float(model.predict(features)[0])
    predicted_score = round(max(0.0, min(100.0, predicted_score)), 2)
    result = "Pass" if predicted_score >= 40 else "Fail"
    return {
        "predicted_score": predicted_score,
        "predicted_result": result,
        "pass_mark": 40
    }
