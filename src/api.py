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
        "recourseJobId": job_id, 
        "message": "Decision generated. Recourse optimization queued." if job_id else "Decision generated."
    }

@app.get("/api/v1/metrics/sensitivity")
def get_sensitivity_metrics():
    """Returns the pre-computed drift sensitivity curve data."""
    try:
        if os.path.exists('results/Table_03_drift_magnitude_sweep.csv'):
            df = pd.read_csv('results/Table_03_drift_magnitude_sweep.csv')
            return df.to_dict(orient='records')
        elif os.path.exists('results/sensitivity_metrics.json'):
            with open('results/sensitivity_metrics.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Sensitivity metrics not found"}

@app.get("/api/v1/metrics/failure-cases")
def get_failure_cases():
    """Returns representative thin-file failure case payloads."""
    try:
        with open('results/failure_cases.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Failure cases not generated yet"}

class FeedbackSubmission(BaseModel):
    applicationId: str
    reviewerId: str
    applicantSubgroup: str
    actionabilityScore: int
    plausibilityScore: int
    verdict: str
    reviewerNotes: str

@app.post("/api/v1/feedback")
def submit_feedback(fb: FeedbackSubmission):
    """Stores expert underwriter ratings in results/expert_feedback.csv."""
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/expert_feedback.csv'
    new_row = pd.DataFrame([{
        'application_id': fb.applicationId,
        'reviewer_id': fb.reviewerId,
        'applicant_subgroup': fb.applicantSubgroup,
        'actionability_score': fb.actionabilityScore,
        'plausibility_score': fb.plausibilityScore,
        'verdict': fb.verdict,
        'reviewer_notes': fb.reviewerNotes
    }])
    if os.path.exists(csv_path):
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(csv_path, index=False)
    return {"status": "success", "message": "Feedback recorded."}

@app.get("/api/v1/feedback")
def get_feedback():
    """Retrieves all submitted expert underwriter reviews."""
    csv_path = 'results/expert_feedback.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_dict(orient='records')
    return []

