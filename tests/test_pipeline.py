import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import assign_subgroups, simulate_drift_split, load_and_clean_data
from src.models import train_baseline_models, train_retrained_models
from src.evaluation import (
    run_significance_tests, compute_invalidation_rates,
    compute_classification_metrics, compute_action_cost
)
from src.explainers import compute_attribution_rank_correlations
from fastapi.testclient import TestClient
from src.api import app

def test_subgroup_assignment():
    df = pd.DataFrame({
        'MSinceOldestTradeOpen': [10, 50, 100],
        'NumTotalTrades': [2, 10, 20]
    })
    df_assigned = assign_subgroups(df, quantile_cutoff=0.25)
    assert df_assigned.loc[0, 'Subgroup'] == 'thin-file'
    assert df_assigned.loc[2, 'Subgroup'] == 'thick-file'

def test_significance_tests():
    np.random.seed(42)
    subgroups = pd.Series(['thin-file']*50 + ['thick-file']*50)
    invalidation_flags = np.array([True]*40 + [False]*10 + [True]*10 + [False]*40)
    
    air, sir = compute_invalidation_rates(invalidation_flags, subgroups)
    assert sir['thin-file'] == 0.8
    assert sir['thick-file'] == 0.2
    
    results = run_significance_tests(invalidation_flags, subgroups)
    assert results['significant_difference'] is True
    assert results['z_test_pvalue'] < 0.05
    assert results['fisher_exact_pvalue'] < 0.05
    assert results['cohens_h_effect_size'] is not None

def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 'fpr' in metrics
    assert 'fnr' in metrics
    assert metrics['accuracy'] == 0.8

def test_action_cost_calculation():
    X_orig = pd.DataFrame({'NetFractionRevolvingBurden': [80.0, 90.0]})
    X_cf = pd.DataFrame({'NetFractionRevolvingBurden': [40.0, 45.0]})
    costs = compute_action_cost(X_orig, X_cf, ['NetFractionRevolvingBurden'])
    assert len(costs) == 2
    assert costs[0] > 0

def test_rank_correlation():
    shap_0 = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    shap_1 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
    res = compute_attribution_rank_correlations(shap_0, shap_1)
    assert 'mean_spearman' in res
    assert 'attribution_stability_cosine' in res

def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_feedback_endpoints():
    client = TestClient(app)
    payload = {
        "applicationId": "TEST-APP-001",
        "reviewerId": "TEST-REV",
        "applicantSubgroup": "thin-file",
        "actionabilityScore": 4,
        "plausibilityScore": 3,
        "verdict": "ACCEPT",
        "reviewerNotes": "Automated unit test review"
    }
    post_res = client.post("/api/v1/feedback", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "success"
    
    get_res = client.get("/api/v1/feedback")
    assert get_res.status_code == 200
    assert isinstance(get_res.json(), list)

def test_ml_benchmarks_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/ml/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "summaryKPIs" in data
    assert "modelPerformance" in data
    assert "subgroupFairness" in data
    assert data["summaryKPIs"]["thinFileRecallDrop"] == "-31.7%"

def test_ml_diagnostic_curves_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/ml/diagnostic-curves")
    assert response.status_code == 200
    data = response.json()
    assert "rocCurves" in data
    assert "prCurves" in data
    assert "confusionMatrices" in data
    assert "subgroupErrorRates" in data
    assert len(data["rocCurves"]) > 5
    assert len(data["prCurves"]) > 5

