from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(title="Credit Model ML Service")

# Load the trained models
MODEL_PATH = "results/models/xgb_t1.pkl"
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}. {e}")

class ApplicantData(BaseModel):
    ExternalRiskEstimate: float
    MSinceOldestTradeOpen: float
    MSinceMostRecentTradeOpen: float
    AverageMInFile: float
    NumSatisfactoryTrades: float
    NumTrades60Ever2DerogPubRec: float
    NumTrades90Ever2DerogPubRec: float
    PercentTradesNeverDelq: float
    MSinceMostRecentDelq: float
    MaxDelq2PublicRecLast12M: float
    MaxDelqEver: float
    NumTotalTrades: float
    NumTradesOpeninLast12M: float
    PercentInstallTrades: float
    MSinceMostRecentInqexcl7days: float
    NumInqLast6M: float
    NumInqLast6Mexcl7days: float
    NetFractionRevolvingBurden: float
    NetFractionInstallBurden: float
    NumRevolvingTradesWBalance: float
    NumInstallTradesWBalance: float
    NumBank2NatlTradesWHighUtilization: float
    PercentTradesWBalance: float

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: ApplicantData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict using the T1 model (post-drift)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] # Probability of Approval (1)
    
    status = "Approved" if prediction == 1 else "Denied"
    
    return {
        "prediction": int(prediction),
        "status": status,
        "approval_probability": float(probability)
    }

# In a full implementation, we'd add endpoints for /shap and /recourse here
