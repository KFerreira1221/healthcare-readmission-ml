from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/readmission_model.joblib")

class ModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first with: python src/train.py"
            )
        self.model = joblib.load(MODEL_PATH)

    def predict(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])
        prob = float(self.model.predict_proba(df)[:, 1][0])
        pred = int(prob >= 0.5)
        return {"readmission_probability": prob, "readmitted_30d": pred}