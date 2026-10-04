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

from fastapi import BackgroundTasks
import time
import uuid

class LoanEvaluationRequest(BaseModel):
    applicantId: str
    features: dict
    subgroup: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

def run_dice_optimization(job_id: str, applicant_id: str, features: dict):
    """
    Simulates a heavy DiCE optimization task running on a Celery/Redis queue worker.
    In a real environment, this would compute the counterfactuals and post a webhook
    back to the Java Spring Boot API Gateway.
    """
    print(f"[Worker] Starting DiCE optimization for job {job_id} (Applicant {applicant_id})")
    time.sleep(3) # Simulate heavy DiCE latency (3 seconds)
    print(f"[Worker] Finished DiCE for job {job_id}. Action cost computed.")
    # Here it would make a POST request to Java API: /api/v1/webhooks/recourse

@app.post("/predict")
def predict(request: LoanEvaluationRequest, background_tasks: BackgroundTasks):
    """
    Production-ready endpoint handling the real-time latency budget.
    Returns the immediate ML prediction instantly (<100ms) and queues the heavy 
    DiCE optimization (seconds) to a background worker.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input to DataFrame
    df = pd.DataFrame([request.features])
    
    # Predict using the T1 model (post-drift)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] # Probability of Approval (1)
    
    status = "Approved" if prediction == 1 else "Denied"
    
    # If denied, queue the DiCE counterfactual generation to a background task
    # to avoid blocking the HTTP thread (solving the latency budget gap)
    job_id = None
    if status == "Denied":
        job_id = str(uuid.uuid4())
        background_tasks.add_task(run_dice_optimization, job_id, request.applicantId, request.features)
    
    return {
        "applicantId": request.applicantId,
        "decision": status,
        "approvalProbability": float(probability),
        "recourseJobId": job_id, # Frontend/Java can poll this or wait for webhook
        "message": "Decision generated. Recourse optimization queued." if job_id else "Decision generated."
    }

