from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female", examples=[1])
    cp: int = Field(..., ge=0, le=4, description="chest pain type", examples=[1])
    trestbps: float = Field(..., description="resting blood pressure (mm Hg)", examples=[145])
    chol: float = Field(..., description="serum cholesterol (mg/dl)", examples=[233])
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120 mg/dl", examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., description="max heart rate achieved", examples=[150])
    exang: int = Field(..., ge=0, le=1, description="exercise induced angina", examples=[0])
    oldpeak: float = Field(..., examples=[2.3])
    slope: int = Field(..., ge=0, le=3, examples=[3])
    ca: float = Field(..., ge=0, le=4, description="number of major vessels", examples=[0])
    thal: float = Field(..., examples=[6])


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str
    confidence: float = Field(..., description="model probability of the predicted class")
