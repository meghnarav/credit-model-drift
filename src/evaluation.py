import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_recourse_invalidation(model_t1, df_recourse, features):
    """
    Evaluates original recourse points against retrained models f1(x*).
    df_recourse contains the counterfactual instances (x*).
    Returns boolean array of invalidation: True if f1(x*) == 0.
    """
    X_star = df_recourse[features]
    predictions_t1 = model_t1.predict(X_star)
    
    # Invalidation occurs if the prediction flips back to 0 (Denied) under T1
    invalidation_flags = (predictions_t1 == 0)
    return invalidation_flags
def compute_action_cost(X_original, X_recourse, features_to_vary):
    """
    Computes the L1 distance (action cost) between the original instances and the counterfactuals,
    normalized by the standard deviation of the features to ensure fair weighting.
    """
    cost_metrics = []
    
    # Calculate std dev for normalization based on the original data provided
    std_devs = X_original[features_to_vary].std()
    # Avoid division by zero
    std_devs = std_devs.replace(0, 1)
    
    for i in range(len(X_original)):
        orig = X_original[features_to_vary].iloc[i]
        recourse = X_recourse[features_to_vary].iloc[i]
        
        # L1 (Manhattan) distance, normalized
        l1_distance = np.sum(np.abs(orig - recourse) / std_devs)
        cost_metrics.append(float(l1_distance))
        
    return np.array(cost_metrics)

def compute_invalidation_rates(invalidation_flags, subgroups=None):
    """
    Computes Aggregate Invalidation Rate (AIR) and Subgroup Invalidation Rate (SIR).
    """
    air = np.mean(invalidation_flags)
    
    sir = {}
    if subgroups is not None:
        for group in subgroups.unique():
            idx = (subgroups == group)
            if np.sum(idx) > 0:
                sir[group] = np.mean(invalidation_flags[idx])
            else:
                sir[group] = np.nan
                
    return air, sir

def run_significance_tests(
    invalidation_flags, 
    subgroups, 
    group_a='thin-file', 
    group_b='thick-file',
    n_bootstrap=1000,
    seed=42
):
    """
    Runs a two-proportion Z-test and Fisher's exact test with:
    - Exact sample counts
    - Effect size (Cohen's h)
    - Non-parametric bootstrap 95% Confidence Interval for difference in proportions
    - Sample size warning if n < 30
    """
    invalidation_flags = np.array(invalidation_flags)
    subgroups = np.array(subgroups)
    
    idx_a = (subgroups == group_a)
    idx_b = (subgroups == group_b)
    
    count_a = int(np.sum(invalidation_flags[idx_a]))
    nobs_a = int(np.sum(idx_a))
    
    count_b = int(np.sum(invalidation_flags[idx_b]))
    nobs_b = int(np.sum(idx_b))
    
    prop_a = count_a / nobs_a if nobs_a > 0 else np.nan
    prop_b = count_b / nobs_b if nobs_b > 0 else np.nan
    
    sample_size_warning = None
    if nobs_a < 30 or nobs_b < 30:
        sample_size_warning = f"Warning: Small sample size (n_{group_a}={nobs_a}, n_{group_b}={nobs_b} < 30). Z-test asymptotic assumptions may be fragile; rely on Fisher's exact test."
    
    counts = np.array([count_a, count_b])
    nobs = np.array([nobs_a, nobs_b])
    
    pval_z, pval_fisher = np.nan, np.nan
    cohens_h = np.nan
    ci_lower, ci_upper = np.nan, np.nan
    
    if nobs_a > 0 and nobs_b > 0:
        try:
            stat_z, pval_z = proportions_ztest(counts, nobs)
        except Exception:
            pval_z = np.nan
            
        # Fisher's exact test
        table = [
            [count_a, nobs_a - count_a],
            [count_b, nobs_b - count_b]
        ]
        try:
            res_fisher = stats.fisher_exact(table)
            pval_fisher = res_fisher[1]
        except Exception:
            pval_fisher = np.nan
            
        # Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        p1 = min(max(prop_a, 1e-4), 1 - 1e-4)
        p2 = min(max(prop_b, 1e-4), 1 - 1e-4)
        cohens_h = float(2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))
        
        # Bootstrap 95% CI for difference (p_a - p_b)
        rng = np.random.default_rng(seed)
        diffs = []
        flags_a = invalidation_flags[idx_a]
        flags_b = invalidation_flags[idx_b]
        for _ in range(n_bootstrap):
            b_sample_a = rng.choice(flags_a, size=nobs_a, replace=True)
            b_sample_b = rng.choice(flags_b, size=nobs_b, replace=True)
            diffs.append(np.mean(b_sample_a) - np.mean(b_sample_b))
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        
    return {
        'group_a': group_a,
        'group_b': group_b,
        'count_a': count_a,
        'nobs_a': nobs_a,
        'prop_a': float(prop_a) if not np.isnan(prop_a) else None,
        'count_b': count_b,
        'nobs_b': nobs_b,
        'prop_b': float(prop_b) if not np.isnan(prop_b) else None,
        'z_test_pvalue': float(pval_z) if not np.isnan(pval_z) else None,
        'fisher_exact_pvalue': float(pval_fisher) if not np.isnan(pval_fisher) else None,
        'cohens_h_effect_size': float(cohens_h) if not np.isnan(cohens_h) else None,
        'bootstrap_ci_95': [ci_lower, ci_upper] if not np.isnan(ci_lower) else None,
        'warning': sample_size_warning,
        'significant_difference': bool((pval_z < 0.05) or (pval_fisher < 0.05))
    }


def output_results(air, sir, tests, output_dir='.'):
    """
    Outputs results into a structured JSON summary and LaTeX-ready table format.
    """
    results = {
        'AIR': float(air),
        'SIR': {k: float(v) for k, v in sir.items()},
        'Tests': tests
    }
    
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # LaTeX table
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Aggregate Invalidation Rate (AIR) & {air:.2%} \\\\
SIR (Thin-file) & {sir.get('thin-file', np.nan):.2%} \\\\
SIR (Thick-file) & {sir.get('thick-file', np.nan):.2%} \\\\
Z-test p-value & {tests['z_test_pvalue']:.4f} \\\\
Fisher Exact p-value & {tests['fisher_exact_pvalue']:.4f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Counterfactual Invalidation Rates and Significance Tests}}
\\end{{table}}
"""
    with open(f'{output_dir}/evaluation_results.tex', 'w') as f:
        f.write(latex_table)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes standard credit-risk ML metrics:
    Accuracy, Precision, Recall, F1-score, ROC-AUC, FPR (False Positive Rate), FNR (False Negative Rate).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = np.nan
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
            
    # Confusion matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fpr, fnr = 0.0, 0.0
        
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc) if not np.isnan(auc) else None,
        'fpr': float(fpr),
        'fnr': float(fnr)
    }

def evaluate_subgroup_ml_fairness(df, features, model, y_true_col='RiskPerformance', subgroup_col='Subgroup'):
    """
    Evaluates ML performance separately for thin-file, thick-file, and other subgroups.
    Computes subgroup Recall, FPR, FNR, and detects disparate degradation.
    """
    results = {}
    subgroups = df[subgroup_col].dropna().unique()
    
    for grp in subgroups:
        sub_df = df[df[subgroup_col] == grp]
        if len(sub_df) == 0:
            continue
            
        X = sub_df[features]
        y_true = sub_df[y_true_col]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics['sample_size'] = int(len(sub_df))
        results[grp] = metrics
        
    return results

def run_ml_drift_comparison(lr_t0, xgb_t0, lr_t1, xgb_t1, df_t0, df_t1, features, target='RiskPerformance'):
    """
    Full Stage 1 ML experiment:
    Evaluates LR and XGBoost at:
    1. Baseline (T0 test data evaluated with f0)
    2. Drifted (T1 test data evaluated with un-retrained f0)
    3. Retrained (T1 test data evaluated with retrained f1)
    Across overall population and thin-file vs thick-file subgroups.
    """
    models = {
        'LogisticRegression': {'baseline': lr_t0, 'retrained': lr_t1},
        'XGBoost': {'baseline': xgb_t0, 'retrained': xgb_t1}
    }
    
    comparison = {}
    for name, m_dict in models.items():
        m_t0 = m_dict['baseline']
        m_t1 = m_dict['retrained']
        
        # 1. Baseline performance on T0
        y_pred_t0 = m_t0.predict(df_t0[features])
        y_prob_t0 = m_t0.predict_proba(df_t0[features])[:, 1] if hasattr(m_t0, 'predict_proba') else None
        overall_t0 = compute_classification_metrics(df_t0[target], y_pred_t0, y_prob_t0)
        subgroup_t0 = evaluate_subgroup_ml_fairness(df_t0, features, m_t0, target)
        
        # 2. Drifted performance on T1 using unretrained model f0
        y_pred_drift = m_t0.predict(df_t1[features])
        y_prob_drift = m_t0.predict_proba(df_t1[features])[:, 1] if hasattr(m_t0, 'predict_proba') else None
        overall_drift = compute_classification_metrics(df_t1[target], y_pred_drift, y_prob_drift)
        subgroup_drift = evaluate_subgroup_ml_fairness(df_t1, features, m_t0, target)
        
        # 3. Retrained performance on T1 using retrained model f1
        y_pred_retrained = m_t1.predict(df_t1[features])
        y_prob_retrained = m_t1.predict_proba(df_t1[features])[:, 1] if hasattr(m_t1, 'predict_proba') else None
        overall_retrained = compute_classification_metrics(df_t1[target], y_pred_retrained, y_prob_retrained)
        subgroup_retrained = evaluate_subgroup_ml_fairness(df_t1, features, m_t1, target)
        
        comparison[name] = {
            'overall': {
                'baseline_t0': overall_t0,
                'drifted_t1_f0': overall_drift,
                'retrained_t1_f1': overall_retrained
            },
            'subgroups': {
                'baseline_t0': subgroup_t0,
                'drifted_t1_f0': subgroup_drift,
                'retrained_t1_f1': subgroup_retrained
            }
        }
        
    return comparison

