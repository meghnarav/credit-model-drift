# Credit Model Drift: Fragility of Algorithmic Recourse

This repository contains the end-to-end experimental framework and interactive dashboard for the research paper: *"On the Fragility of Algorithmic Recourse: Evaluating Subgroup-Differentiated Counterfactual Invalidation Under Credit Model Drift"*.

## Architecture

1.  **Data Engineering & Subgroup Partitioning (`src/data.py`)**: Data ingestion, cleaning (FICO HELOC), stratification (thin-file vs thick-file), and drift simulation.
2.  **Model Training & Attribution (`src/models.py`, `src/explainers.py`)**: Baseline and retrained model training (Logistic Regression, XGBoost), feature attributions (SHAP), and recourse generation (DiCE).
3.  **Recourse Stability & Significance Testing (`src/evaluation.py`)**: Evaluation of recourse invalidation rates (AIR, SIR) and statistical significance testing.
4.  **Interactive Verification Dashboard (`src/dashboard.py`)**: Streamlit-based UI with a Bento Grid layout for analyzing applicant profiles, attributions, recourse options, and reliability metrics.

## Setup

```bash
pip install -r requirements.txt
```
