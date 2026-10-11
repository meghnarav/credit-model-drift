# Presentation Slide Outline: Credit Risk Prediction Under Drift & Subgroup Fairness

## Slide 1: Title & Executive Summary
- Title: On the Fragility of Algorithmic Recourse Under Model Drift
- Subtitle: Evaluating Robustness, Subgroup Fairness, and Explanation Invalidation on FICO HELOC
- Presenter & Course Information

## Slide 2: Motivation & Problem Statement
- Credit Scoring Non-Stationarity: Macroeconomic cycles shift borrower credit utilization
- The Recourse Dilemma: Why actionable advice given today breaks tomorrow
- Core Research Question: Does model retraining silently invalidate counterfactual recourse, and who bears the cost?

## Slide 3: Dataset & Experimental Design
- FICO HELOC Dataset ($N = 10,459$) & Special Code Sanitization (-9, -8, -7)
- Subgroup Stratification: Operationalizing Thin-File vs. Thick-File Borrowers ($Q_{25}$ credit history & trade volume)
- Controlled Covariate Drift Simulation: Macroeconomic shift on `NetFractionRevolvingBurden` ($\Delta \in [0.05, 0.35]$)
- Multi-Seed Experimental Protocol ($5$ seeds, mean $\pm$ std, $95\%$ bootstrap CIs)

## Slide 4: Base Machine Learning Results (Stage 1)
- Logistic Regression vs. XGBoost: Linear benchmark vs. Non-linear Gradient Boosted Trees
- The Resilience Paradox: Linear models resist covariate shift; trees suffer severe degradation before retraining
- Subgroup Recall Collapse: Thin-file recall plunges from $98.1\%$ to $34.8\%$ under drift ($63.3\%$ drop)
- Error Asymmetry: Surge in False Negative Rates (denying creditworthy thin-file applicants)

## Slide 5: Explainable AI: Feature Attribution Drift (SHAP)
- Global Feature Importance: $f_0$ baseline vs. $f_1$ retrained
- Attribution Rank Correlations: Spearman $\rho$ and Kendall $\tau$ across model revisions
- Cosine Stability Metric: Tracking shifting decision weights under economic stress

## Slide 6: Algorithmic Recourse Invalidation (DiCE)
- Generating Actionable Counterfactuals for Denied Applicants ($f_0(x^*) = 1$)
- The Invalidation Phenomenon: Evaluating $f_1(x^*)$ on the updated policy
- Empirical Disparity: Aggregate Invalidation Rate ($80.0\%$) vs. Subgroup Invalidation Rate ($66.7\%$ thin-file vs. $60.0\%$ thick-file)
- Normalized $L_1$ Action Cost: Quantifying the increased effort barrier to re-qualification

## Slide 7: Statistical Significance & Hypothesis Testing
- Two-Proportion Z-Test & Fisher's Exact Test ($p < 0.05$)
- Effect Size: Cohen's $h$ & Non-Parametric Bootstrap $95\%$ Confidence Intervals
- Small-Sample Guardrails and Sensitivity to Subgroup Sample Size

## Slide 8: Multi-Dimensional Sensitivity & Ablation Suite
- Drift Magnitude Curve: Invalidation rates vs. $\Delta$ perturbation
- Alternate Shift Regimes: Label/Prior shift and tenure feature shift (`MSinceOldestTradeOpen`)
- Tornado Ranking Chart: Ranking the dominant drivers of the thin-vs-thick disparity gap
- Interaction Heatmaps: Hyperparameter depth vs. learning rate sensitivity

## Slide 9: Failure Case Profiling & Boundary-Jitter
- Decision Boundary Proximity Analysis
- Representative SHAP Waterfall Case Studies across Subgroups
- Dominant Invalidation Mechanisms: Debt capacity shifts vs. immutable credit maturity penalties

## Slide 10: Human-in-the-Loop Expert Feedback (IML)
- Auditing Recourse with Credit Committee Reviewers ($3$ Underwriter Personas)
- Structured Scoring: Actionability ($1-5$), Plausibility ($1-5$), and Verdict Distribution
- Disparity in Expert Acceptance: Why human underwriters reject $55\%$ of thin-file algorithmic advice

## Slide 11: Production Microservices & Deployment Architecture
- High-Throughput ML Engine (FastAPI) with Async Optimization Queue
- Financial Audit Gateway (Spring Boot 3 + PostgreSQL JPA Schema)
- Interactive Bento Grid Dashboard (Next.js 15 + Neo-Brutalist Design)
- Container Orchestration & Cloud Healthcheck Verification

## Slide 12: Limitations, Failures, & Future Research
- Known Limitations: Synthetic covariate shift vs. true longitudinal temporal cohorts
- DiCE Optimization Latency & Convexity Challenges
- Path Forward: Shift-robust recourse objectives and proactive delta-margin training
