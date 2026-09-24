# Research Specification: FICO HELOC Data Engineering & Recourse Constraints

---

## 1. Executive Summary & Research Goal

* **Statement:** "On the Fragility of Algorithmic Recourse: Evaluating Subgroup-Differentiated Counterfactual Invalidation Under Credit Model Drift."
* **Objective:** Formalize the mathematical and structural constraints required to evaluate whether counterfactual explanations generated under a baseline credit scoring model ($T_0$) remain valid after macroeconomic and covariate shift ($T_1$), specifically measuring disparity across thin-file and thick-file borrower subgroups.

---

## 2. FICO HELOC Negative-Code Handling Specification

The raw dataset (`heloc_dataset.csv`) contains special code values that represent missing data, specific application statuses, or structural non-applicability. These must be sanitized and handled prior to model training:

| Special Code | Value | Semantic Meaning | Preprocessing Strategy |
| --- | --- | --- | --- |
| **$-9$** | No Bureau Record / Insufficient Data | Applicant has no historical record for the feature. | Imputed via domain-specific default or isolated category flag depending on model type. |
| **$-8$** | Not Applicable | The credit line or trade attribute does not apply to the borrower's file. | Treated as a structural zero or handled through missingness indicators. |
| **$-7$** | Condition Not Met / Too Few Trades | Trade count or credit history duration falls below reporting thresholds. | Mapped to $0$ for counts/durations or explicitly separated for subgroup stratification. |

---

## 3. Subgroup Classification Rules

Borrowers are stratified into vulnerability tiers based on structural credit history metrics to evaluate disparate impact during model drift:

* **Thin-File Subgroup:**

$$(\text{MSinceOldestTradeOpen} \le \text{25th Percentile}) \land (\text{NumTotalTrades} \le \text{25th Percentile})$$


* **Thick-File Subgroup:**

$$(\text{MSinceOldestTradeOpen} \ge \text{50th Percentile}) \land (\text{NumTotalTrades} \ge \text{50th Percentile})$$



---

## 4. DiCE Feature-Dictionary & Recourse Constraints

To ensure generated counterfactuals ($x^*$) represent realistic, legally compliant, and financially actionable steps for applicants, the DiCE constraint dictionary is defined as follows:

```python
dice_constraints = {
    "immutable_features": [
        "MSinceOldestTradeOpen",  # Time cannot be artificially reversed
        "NumTotalTrades"          # Historical volume cannot be instantly altered by user action
    ],
    "actionable_features": [
        "NetFractionRevolvingBurden", # Can be improved by paying down revolving balances
        "NumSatisfactoryTrades",      # Can be improved over time via consistent payments
        "ExternalRiskEstimate"        # Indirectly actionable via credit score hygiene
    ],
    "target_flipping_rule": "f0(x_star) == 1",  # Must successfully flip denial to approval
    "permitted_range_deviations": {
        "NetFractionRevolvingBurden": (0.0, 100.0)
    }
}
```

---

## 5. Mathematical Formulations & Statistical Tests

* **Aggregate Invalidation Rate (AIR):**

$$\text{AIR} = \frac{1}{N_{\text{denied}}} \sum_{i=1}^{N_{\text{denied}}} \mathbb{I}(f_1(x_i^*) = 0)$$


* **Subgroup Invalidation Rate (SIR):**

$$\text{SIR}_{g} = \frac{1}{N_{g}} \sum_{i \in g} \mathbb{I}(f_1(x_i^*) = 0) \quad \text{for } g \in \{\text{thin-file}, \text{thick-file}\}$$


* **Hypothesis Testing:**
A two-proportion Z-test and Fisher's exact test are executed to test the null hypothesis $H_0: \text{SIR}_{\text{thin}} = \text{SIR}_{\text{thick}}$ against the alternative hypothesis of differential vulnerability under drift ($p < 0.05$).
