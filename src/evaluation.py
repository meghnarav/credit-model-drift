import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats
import json

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

def run_significance_tests(invalidation_flags, subgroups, group_a='thin-file', group_b='thick-file'):
    """
    Runs a two-proportion Z-test and Fisher's exact test to assess whether the difference 
    in invalidation rates between subgroups is statistically significant (p < 0.05).
    """
    idx_a = (subgroups == group_a)
    idx_b = (subgroups == group_b)
    
    count_a = np.sum(invalidation_flags[idx_a])
    nobs_a = np.sum(idx_a)
    
    count_b = np.sum(invalidation_flags[idx_b])
    nobs_b = np.sum(idx_b)
    
    # 1. Z-test for proportions
    counts = np.array([count_a, count_b])
    nobs = np.array([nobs_a, nobs_b])
    
    if nobs_a > 0 and nobs_b > 0:
        stat_z, pval_z = proportions_ztest(counts, nobs)
        
        # 2. Fisher's exact test
        # Contingency table: [[invalidated_A, valid_A], [invalidated_B, valid_B]]
        table = [
            [count_a, nobs_a - count_a],
            [count_b, nobs_b - count_b]
        ]
        res_fisher = stats.fisher_exact(table)
        pval_fisher = res_fisher[1]
    else:
        stat_z, pval_z, pval_fisher = np.nan, np.nan, np.nan
        
    return {
        'z_test_pvalue': pval_z,
        'fisher_exact_pvalue': pval_fisher,
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
