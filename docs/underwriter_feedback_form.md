# Underwriter Recourse Audit & Review Form
**Project:** Credit Risk Prediction Under Drift — Algorithmic Recourse Evaluation  
**Auditor Instructions:** Review the applicant's current profile, the prescribed algorithmic recourse plan, and the post-drift model decision. Score each case on Actionability (1-5), Plausibility (1-5), and record your final Credit Committee Verdict.

---

### Case Evaluation Template

| Field | Reviewer 1 (Senior Risk Officer) | Reviewer 2 (Compliance Lead) | Reviewer 3 (Credit Underwriter) |
|:---|:---|:---|:---|
| **Auditor ID** | `REV-001` | `REV-002` | `REV-003` |
| **Applicant ID** | `APP-9824` | `APP-9824` | `APP-9824` |
| **Applicant Subgroup** | [ ] Thin-File &nbsp; [ ] Thick-File | [ ] Thin-File &nbsp; [ ] Thick-File | [ ] Thin-File &nbsp; [ ] Thick-File |
| **Prescribed Recourse Target** | Reduce Revolving Burden: 85% → 45% | Reduce Revolving Burden: 85% → 45% | Reduce Revolving Burden: 85% → 45% |
| **Actionability Score (1-5)**<br>*(1=Impossible, 5=Easily Actionable)* | [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4 &nbsp; [ ] 5 | [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4 &nbsp; [ ] 5 | [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4 &nbsp; [ ] 5 |
| **Economic Plausibility (1-5)**<br>*(1=Implausible, 5=Highly Realistic)* | [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4 &nbsp; [ ] 5 | [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4 &nbsp; [ ] 5 | [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4 &nbsp; [ ] 5 |
| **Final Underwriter Verdict** | [ ] ACCEPT &nbsp; [ ] MODIFY &nbsp; [ ] REJECT | [ ] ACCEPT &nbsp; [ ] MODIFY &nbsp; [ ] REJECT | [ ] ACCEPT &nbsp; [ ] MODIFY &nbsp; [ ] REJECT |
| **Required Override Notes** | __________________________________________ | __________________________________________ | __________________________________________ |
| **Date & Signature** | __________________________________________ | __________________________________________ | __________________________________________ |

---

### Guidelines for Scoring
- **Actionability:** Can a consumer realistically execute this perturbation within 6-12 months without financial distress?
- **Plausibility:** Does the counterfactual vector conform to joint financial distributions (e.g. paying down 40% burden without new trade lines)?
- **Compliance Verdict:**
  - `ACCEPT`: Recourse advice is sound and approved for customer delivery.
  - `MODIFY`: Requires manual override adjustment before applicant notification.
  - `REJECT`: Invalidation risk exceeds safety threshold; advice must be withheld.
