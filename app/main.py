from fastapi import FastAPI
from app.schemas import ReadmissionFeatures, PredictionResponse
from app.predict import ModelService

app = FastAPI(title="Readmission Risk Predictor", version="1.0.0")
svc = ModelService()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: ReadmissionFeatures):
    return svc.predict(features.model_dump())