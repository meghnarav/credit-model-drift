# Literature Review: Comparison Table Template

This empty template provides a standardized matrix for comparing prior academic literature and industry tools against this project's research framework.

| Tool / Paper Citation | Predictive Task | Drift Handled (Covariate / Label / Concept) | Subgroup Fairness Evaluated? | XAI Method Used (SHAP / LIME / etc.) | Recourse / Counterfactual Evaluated? | Key Limitations & Invalidation Observed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **This Study (Credit Model Drift)** | FICO HELOC Credit Default Risk | Covariate Shift on Revolving Debt ($\Delta \in [0.05, 0.35]$) & Prior Shift | Thin-file vs. Thick-file (Q20/Q25/Q30 cutoffs) | TreeSHAP & LinearSHAP (stability & rank correlation) | DiCE Counterfactuals ($k \in [1, 3]$, actionable constraints) | Recourse invalidation reaches 66.7% for thin-file applicants vs 20.0% thick-file ($p < 0.05$); $L_1$ action cost increases. |
| *Template Row 1* | | | | | | |
| *Template Row 2* | | | | | | |
| *Template Row 3* | | | | | | |
| *Template Row 4* | | | | | | |
| *Template Row 5* | | | | | | |
| *Template Row 6* | | | | | | |

---

### Dimensions Legend
1. **Tool / Paper Citation:** Author, Year, Title, and Venue.
2. **Predictive Task:** Dataset domain (e.g. Credit risk, Mortgage lending, Loan default).
3. **Drift Handled:** Type of distribution shift modeled (Covariate $P(X)$, Prior/Label $P(Y)$, Concept $P(Y \mid X)$).
4. **Subgroup Fairness:** Demographics or financial footprint subgroups (e.g. Thin-file, Age, Race, Protected classes).
5. **XAI Method:** Post-hoc feature attribution algorithms analyzed.
6. **Recourse / Counterfactual:** Model-agnostic or optimization-based recourse framework (DiCE, Wachter, AR).
7. **Key Limitations:** Invalidation rate observed, non-actionability, distance metric, computational complexity.
