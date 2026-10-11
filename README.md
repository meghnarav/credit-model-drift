# Credit Risk Prediction Under Model Drift: Evaluating Robustness and Subgroup Fairness
> **Extended into Explainable AI (XAI):** *On the Fragility of Algorithmic Recourse: Evaluating Subgroup-Differentiated Counterfactual Invalidation Under Credit Model Drift*

---

## 📌 Executive Summary & Project Progression

This research project investigates the robustness, subgroup fairness, and algorithmic recourse stability of machine learning models deployed for credit scoring under non-stationary distributions.

The project is structured in two tightly integrated phases:
1. **Stage 1: Foundational Machine Learning & Subgroup Fairness Study**
   - Evaluates credit-risk classification models under baseline and controlled macroeconomic drift conditions.
   - Quantifies whether model performance deteriorates uniformly or disproportionately harms vulnerable subgroups (particularly **thin-file** applicants with short credit histories and few trade lines).
2. **Stage 2: Explainable AI (XAI) & Counterfactual Recourse Invalidation Extension**
   - Evaluates whether feature attributions (SHAP) and actionable recommendations given to rejected applicants (DiCE counterfactual recourse) remain valid once models are retrained ($f_0 \to f_1$).
   - Discloses that recourse invalidation is subgroup-differentiated (e.g., thin-file applicants face a **66.7%** recourse invalidation rate vs. **20.0%** for thick-file applicants, $p < 0.05$).

---

## 🎯 Stage 1: Machine Learning Foundation

### 1. Problem Statement
Credit-risk models are periodically retrained as economic conditions and applicant populations evolve. However, aggregate metrics (such as population-level ROC-AUC or Accuracy) often conceal subgroup-specific degradation. When macro conditions shift, models may degrade silently on applicants with limited credit footprints, amplifying systemic credit denial.

### 2. Dataset: FICO HELOC (Home Equity Line of Credit)
An anonymized dataset containing real-world credit application attributes and risk outcomes:
- **Target (`RiskPerformance`)**: Binary label (`Bad` = 0 [Denied / 90+ days past due], `Good` = 1 [Approved]).
- **Key Features**: `ExternalRiskEstimate`, `MSinceOldestTradeOpen`, `NumTotalTrades`, `NumSatisfactoryTrades`, `NetFractionRevolvingBurden`, delinquencies, and inquiry counts.
- **Data Preprocessing & Special Negative Codes**:
  - `-9`: No Bureau Record (imputed via feature medians for downstream classifiers).
  - `-8`: Not Applicable / Structural Zero (mapped to zero).
  - `-7`: Condition Not Met (mapped to zero counts/durations).

### 3. Subgroup Definitions
Applicants are stratified based on their credit depth and maturity:
- **Thin-File / Short Credit History**: Bottom quartile of credit history length (`MSinceOldestTradeOpen` $\le Q_{25}$) and trade counts (`NumTotalTrades` $\le Q_{25}$).
- **Thick-File**: Applicants at or above median credit history (`MSinceOldestTradeOpen` $\ge Q_{50}$) and trades (`NumTotalTrades` $\ge Q_{50}$).
- **Other**: Intermediate profile applicants.

### 4. Controlled Drift Simulation
- **Covariate Shift**: Biased resampling on `NetFractionRevolvingBurden` simulating tightening macroeconomic credit conditions and surging credit card utilization.
- Baseline distribution ($T_0$) vs. drifted distribution ($T_1$).

### 5. Dual Model Classifiers
- **Logistic Regression**: Scaled linear benchmark with calibrated probabilities.
- **XGBoost**: Non-linear gradient boosted tree classifier capturing complex feature interactions.

### 6. Evaluated ML Metrics
- **Overall**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC across $T_0$, drifted $T_1$ with $f_0$, and retrained $T_1$ with $f_1$.
- **Subgroup Fairness**: Group-level Recall, False Positive Rate (FPR), and False Negative Rate (FNR) to identify disparate error rates.

---

## 🔬 Stage 2: XAI Extension (SHAP & DiCE Recourse)

### 1. Feature Attribution Drift (SHAP)
- Explains model decisions using `shap.LinearExplainer` (Logistic Regression) and `shap.TreeExplainer` (XGBoost).
- Measures whether the relative importance of financial indicators shifts between baseline $f_0$ and retrained $f_1$.

### 2. Algorithmic Recourse Generation (DiCE)
- For applicants denied credit under baseline $f_0(x) = 0$, Diverse Counterfactual Explanations (DiCE) generate minimum-effort changes $x^*$ such that $f_0(x^*) = 1$.
- Enforces actionable features (e.g. lowering `NetFractionRevolvingBurden`, raising `NumSatisfactoryTrades`).

### 3. Subgroup Recourse Invalidation Rate (SIR)
- Measures whether the promised recourse still yields approval under the retrained policy $f_1(x^*)$.
- **Aggregate Invalidation Rate (AIR)**: $\mathbb{E}[ \mathbb{I}(f_1(x^*) = 0) ]$
- **Disparity**: Demonstrates that thin-file applicants experience statistically significant higher invalidation than thick-file applicants.

### 4. Statistical Significance & Action Cost
- **Two-Proportion Z-Test & Fisher's Exact Test**: Confirms subgroup invalidation disparities are statistically significant ($p < 0.05$).
- **Normalized $L_1$ Action Cost Distance**: Measures the distance between original features and counterfactual targets, proving thin-file applicants face higher barriers to re-qualification under drift.

---

## 🏗️ Polyglot Architecture & Microservices

The repository is organized into three production-ready services:

```text
├── ml-engine/ (src/)       # Python 3.11 + FastAPI + XGBoost + Scikit-Learn + SHAP + DiCE-ML
├── api-gateway/            # Java 21 + Spring Boot 3 + PostgreSQL/H2 JPA Audit Trail
├── frontend/               # Next.js 15 + React 19 + Tailwind CSS v4 (Soft Neo-Brutalist Dashboard)
├── data/raw/               # FICO HELOC data storage
├── scripts/                # Automated experiment pipelines (run_pipeline.py)
└── docker-compose.yml      # Orchestration across all services and PostgreSQL 16
```

### 1. `ml-engine` (FastAPI)
- Exposes high-throughput prediction and attribution endpoints (`/api/v1/predict`, `/api/v1/recourse`).
- Asynchronous task worker handling heavy counterfactual optimization without blocking API gateway timeouts.
- Serves drift sensitivity sweeps (`/api/v1/metrics/sensitivity`) and boundary-jitter failure profilers (`/api/v1/metrics/failure-cases`).

### 2. `api-gateway` (Spring Boot 3)
- Enterprise audit trail for financial compliance.
- JPA entities tracking loan applications, counterfactual recourse plans, expert underwriter feedback, and invalidation statuses.

### 3. `frontend` (Next.js 15)
- **Soft Neo-Brutalist Bento Grid Dashboard**:
  - Applicant Profile Card (features & thin/thick file indicator)
  - Real-time Approval Probability & Recourse Reliability Score
  - Interactive SHAP Waterfall / Feature Attribution breakdown
  - Side-by-side DiCE Action Plan with recovery delta indicators
  - Drift Sensitivity Stress-Tester ($\delta \in [0.05, 0.35]$)
  - Underwriter Compliance & Override Modal

---

## 🚀 Running the Experiments & Services

### Quickstart Pipeline
```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Stage 1 ML Benchmark & Stage 2 XAI Recourse Invalidation Pipeline
python scripts/run_pipeline.py
```

### Run Full Stack with Docker Compose
```bash
docker-compose up --build
```
- **Web Dashboard**: `http://localhost:3000`
- **Core Backend**: `http://localhost:8080`
- **ML Engine API**: `http://localhost:8000/docs`

