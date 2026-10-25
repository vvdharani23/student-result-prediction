# Student Result Prediction API

A small FastAPI + scikit-learn deployment practical. It predicts a demonstration score using study hours, attendance, previous score, and assignment score.

> **Important:** `train_model.py` generates synthetic demonstration data. Use a dataset supplied by your instructor if the practical requires a real dataset.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Test prediction

Send a `POST` request to `/predict` with:

```json
{
  "study_hours": 4,
  "attendance": 85,
  "previous_score": 70,
  "assignment_score": 80
}
```

## Endpoints

- `GET /` — API information
- `GET /health` — health check
- `GET /docs` — interactive Swagger documentation
- `POST /predict` — predicted score and Pass/Fail result

## Container

The Dockerfile trains the demonstration model during image build and serves the API on port 8000.

## GitHub Actions / GHCR

The workflow `.github/workflows/publish-ghcr.yml` tests the API and publishes the image to GHCR on pushes to `main`. After the first successful run, check the package visibility in GitHub Packages. For Render, use the actual image URI shown by your GitHub account, for example `ghcr.io/<lowercase-owner>/<repository>:latest`.
