from pydantic import BaseModel, Field

class ReadmissionFeatures(BaseModel):
    encounter_length_hours: float = Field(..., ge=0)
    encounter_class: str

class PredictionResponse(BaseModel):
    readmission_probability: float
    readmitted_30d: int