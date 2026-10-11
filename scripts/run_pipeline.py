import os
import sys
import json
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_and_clean_data, assign_subgroups, simulate_drift_split
from src.models import train_baseline_models, train_retrained_models, save_model
from src.explainers import (
    generate_shap_attributions, setup_dice, generate_recourse,
    compute_attribution_rank_correlations, evaluate_counterfactual_quality
)
from src.evaluation import (
    compute_classification_metrics, evaluate_subgroup_ml_fairness,
    evaluate_recourse_invalidation, compute_invalidation_rates,
    run_significance_tests, compute_action_cost
)
from src.visualizations import (
    plot_roc_and_pr_curves, plot_confusion_matrices, plot_calibration_curves,
    plot_shap_summary_comparison
)
from src.sensitivity_suite import run_comprehensive_sensitivity_suite
from src.failure_analysis import analyze_failure_cases_and_generate_waterfalls
from src.expert_feedback import generate_expert_feedback_artifacts

SEEDS = [42, 101, 202, 303, 404]

def compute_mean_ci(values, confidence=0.95):
    """Computes mean ± std and non-parametric bootstrap 95% confidence interval."""
    val_arr = np.array([v for v in values if not np.isnan(v)])
    if len(val_arr) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "formatted": "N/A"}
    mean_val = float(np.mean(val_arr))
    std_val = float(np.std(val_arr))
    
    # 95% bootstrap CI
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(val_arr, size=len(val_arr), replace=True)) for _ in range(1000)]
    ci_low = float(np.percentile(boot_means, (1 - confidence) / 2 * 100))
    ci_high = float(np.percentile(boot_means, (1 + confidence) / 2 * 100))
    
    return {
        "mean": mean_val,
        "std": std_val,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "formatted": f"{mean_val:.3f} ± {std_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
    }

def main():
    print("="*80)
    print("CREDIT RISK PREDICTION UNDER MODEL DRIFT: COMPLETE EXPERIMENTAL PIPELINE")
    print("="*80)
    
    dataset_path = 'data/raw/heloc_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    # Load and clean dataset
    print("\n[STEP 1/6] Ingesting and Preprocessing FICO HELOC Data...")
    df_clean = load_and_clean_data(dataset_path, imputation_strategy='median')
    df = assign_subgroups(df_clean, quantile_cutoff=0.25)
    features = [c for c in df.columns if c not in ['RiskPerformance', 'Subgroup']]
    
    n_total = len(df)
    n_thin = (df['Subgroup'] == 'thin-file').sum()
    n_thick = (df['Subgroup'] == 'thick-file').sum()
    n_other = (df['Subgroup'] == 'other').sum()
    print(f"  Total: {n_total} applicants | Thin-File: {n_thin} ({n_thin/n_total:.1%}) | Thick-File: {n_thick} ({n_thick/n_total:.1%}) | Other: {n_other}")
    
    # -------------------------------------------------------------
    # 2. Multi-Seed Base Model Experiments (5+ seeds)
    # -------------------------------------------------------------
    print(f"\n[STEP 2/6] Running Base ML Experiments Across {len(SEEDS)} Random Seeds...")
    
    overall_records = []
    subgroup_records = []
    models_last_seed = {}
    df_eval_last_seed = None
    last_predictions = {}
    last_probas = {}
    
    for seed in SEEDS:
        df_t0, df_t1 = simulate_drift_split(df, drift_feature='NetFractionRevolvingBurden', delta=0.25, seed=seed)
        lr_t0, xgb_t0 = train_baseline_models(df_t0, features, seed=seed)
        lr_t1, xgb_t1 = train_retrained_models(df_t1, features, seed=seed)
        
        # Save models on seed 42
        if seed == 42:
            save_model(lr_t0, 'results/models/lr_t0.pkl')
            save_model(lr_t1, 'results/models/lr_t1.pkl')
            save_model(xgb_t0, 'results/models/xgb_t0.pkl')
            save_model(xgb_t1, 'results/models/xgb_t1.pkl')
            models_last_seed = {
                'LR_T0': lr_t0, 'XGB_T0': xgb_t0,
                'LR_T1_unretrained': lr_t0, 'XGB_T1_unretrained': xgb_t0,
                'LR_T1_retrained': lr_t1, 'XGB_T1_retrained': xgb_t1
            }
            df_eval_last_seed = df_t1
            for lbl, m in models_last_seed.items():
                last_predictions[lbl] = m.predict(df_t1[features])
                last_probas[lbl] = m.predict_proba(df_t1[features])[:, 1]
                
        # Evaluate 3 regimes: T0 baseline, T1 unretrained, T1 retrained
        regimes = [
            ('T0_baseline', lr_t0, xgb_t0, df_t0),
            ('T1_drifted_unretrained', lr_t0, xgb_t0, df_t1),
            ('T1_drifted_retrained', lr_t1, xgb_t1, df_t1)
        ]
        
        for regime_name, lr_m, xgb_m, eval_df in regimes:
            for model_name, model_obj in [('LogisticRegression', lr_m), ('XGBoost', xgb_m)]:
                y_true = eval_df['RiskPerformance']
                y_pred = model_obj.predict(eval_df[features])
                y_prob = model_obj.predict_proba(eval_df[features])[:, 1]
                
                m_all = compute_classification_metrics(y_true, y_pred, y_prob)
                overall_records.append({
                    'seed': seed,
                    'model': model_name,
                    'regime': regime_name,
                    'accuracy': m_all['accuracy'],
                    'precision': m_all['precision'],
                    'recall': m_all['recall'],
                    'f1': m_all['f1'],
                    'roc_auc': m_all['roc_auc'],
                    'fpr': m_all['fpr'],
                    'fnr': m_all['fnr']
                })
                
                # Subgroups
                sg_metrics = evaluate_subgroup_ml_fairness(eval_df, features, model_obj)
                for sg_name, metrics in sg_metrics.items():
                    subgroup_records.append({
                        'seed': seed,
                        'model': model_name,
                        'regime': regime_name,
                        'subgroup': sg_name,
                        'sample_size_n': metrics['sample_size'],
                        'recall': metrics['recall'],
                        'fpr': metrics['fpr'],
                        'fnr': metrics['fnr'],
                        'accuracy': metrics['accuracy'],
                        'roc_auc': metrics['roc_auc']
                    })
                    
    df_overall = pd.DataFrame(overall_records)
    df_subgroup = pd.DataFrame(subgroup_records)
    
    # Compute Aggregates: Mean ± Std and 95% CI across 5 seeds
    overall_summary = df_overall.groupby(['model', 'regime']).agg({
        'accuracy': lambda x: compute_mean_ci(x)['formatted'],
        'precision': lambda x: compute_mean_ci(x)['formatted'],
        'recall': lambda x: compute_mean_ci(x)['formatted'],
        'f1': lambda x: compute_mean_ci(x)['formatted'],
        'roc_auc': lambda x: compute_mean_ci(x)['formatted']
    }).reset_index()
    overall_summary.to_csv('results/Table_01_baseline_model_performance.csv', index=False)
    
    subgroup_summary = df_subgroup.groupby(['model', 'regime', 'subgroup']).agg({
        'sample_size_n': 'first',
        'recall': lambda x: compute_mean_ci(x)['formatted'],
        'fpr': lambda x: compute_mean_ci(x)['formatted'],
        'fnr': lambda x: compute_mean_ci(x)['formatted'],
        'roc_auc': lambda x: compute_mean_ci(x)['formatted']
    }).reset_index()
    subgroup_summary.to_csv('results/Table_02_subgroup_fairness_breakdown.csv', index=False)
    
    print("  -> Exported results/Table_01_baseline_model_performance.csv")
    print("  -> Exported results/Table_02_subgroup_fairness_breakdown.csv")
    
    # Generate Base Model Figures
    plot_roc_and_pr_curves(models_last_seed, df_eval_last_seed, features, base_filename="Fig_01_roc_pr_curves")
    plot_confusion_matrices(df_eval_last_seed['RiskPerformance'].values, last_predictions, base_filename="Fig_02_confusion_matrices")
    plot_calibration_curves(df_eval_last_seed['RiskPerformance'].values, last_probas, base_filename="Fig_03_calibration_curves")
    print("  -> Exported Fig_01 (ROC/PR), Fig_02 (Confusion Matrices), Fig_03 (Calibration Curves)")

    # -------------------------------------------------------------
    # 3. XAI Evaluation (SHAP & DiCE Recourse)
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Running XAI Suite: SHAP Stability & DiCE Invalidation...")
    # SHAP Attributions on sample
    df_t0_sample = df.sample(min(200, len(df)), random_state=42)
    xgb_t0_ref = models_last_seed['XGB_T0']
    xgb_t1_ref = models_last_seed['XGB_T1_retrained']
    lr_t0_ref = models_last_seed['LR_T0']
    lr_t1_ref = models_last_seed['LR_T1_retrained']
    
    shap_xgb_t0, _ = generate_shap_attributions(xgb_t0_ref, df_t0_sample[features], model_type='xgboost')
    shap_xgb_t1, _ = generate_shap_attributions(xgb_t1_ref, df_t0_sample[features], model_type='xgboost')
    shap_lr_t0, _ = generate_shap_attributions(lr_t0_ref, df_t0_sample[features], model_type='logreg')
    shap_lr_t1, _ = generate_shap_attributions(lr_t1_ref, df_t0_sample[features], model_type='logreg')
    
    # SHAP Correlations
    corr_xgb = compute_attribution_rank_correlations(shap_xgb_t0, shap_xgb_t1, features)
    corr_lr = compute_attribution_rank_correlations(shap_lr_t0, shap_lr_t1, features)
    
    shap_summary_records = [
        {"model": "XGBoost", "mean_spearman": corr_xgb['mean_spearman'], "mean_kendall": corr_xgb['mean_kendall'], "cosine_stability": corr_xgb['attribution_stability_cosine']},
        {"model": "LogisticRegression", "mean_spearman": corr_lr['mean_spearman'], "mean_kendall": corr_lr['mean_kendall'], "cosine_stability": corr_lr['attribution_stability_cosine']}
    ]
    pd.DataFrame(shap_summary_records).to_csv('results/Table_12_shap_attribution_stability.csv', index=False)
    plot_shap_summary_comparison(shap_xgb_t0, shap_xgb_t1, features, base_filename="Fig_07_shap_global_comparison")
    print("  -> Saved results/Table_12_shap_attribution_stability.csv and Fig_07")
    
    # DiCE Recourse Generation
    print("  -> Generating DiCE Counterfactual Recourse for Denied Applicants...")
    denied_applicants = df_eval_last_seed[xgb_t0_ref.predict(df_eval_last_seed[features]) == 0]
    recourse_sample = denied_applicants.sample(min(30, len(denied_applicants)), random_state=42)
    
    dice_exp = setup_dice(df_t0, xgb_t0_ref, features)
    features_to_vary = ["NetFractionRevolvingBurden", "ExternalRiskEstimate"]
    dice_results = generate_recourse(dice_exp, recourse_sample[features], features_to_vary=features_to_vary, total_CFs=1)
    
    all_cfs = []
    cf_subgroups = []
    valid_query_instances = []
    
    for i, exp in enumerate(dice_results.cf_examples_list):
        if exp.final_cfs_df is not None:
            all_cfs.append(exp.final_cfs_df)
            q_idx = recourse_sample.index[i]
            sg = denied_applicants.loc[q_idx, 'Subgroup']
            cf_subgroups.extend([sg] * len(exp.final_cfs_df))
            valid_query_instances.append(recourse_sample.loc[[q_idx]])
            
    if all_cfs:
        df_all_cfs = pd.concat(all_cfs)
        df_valid_queries = pd.concat(valid_query_instances)
        
        # Invalidation testing
        inv_flags_xgb = evaluate_recourse_invalidation(xgb_t1_ref, df_all_cfs, features)
        inv_flags_lr = evaluate_recourse_invalidation(lr_t1_ref, df_all_cfs, features)
        
        # Normalized L1 Action Cost
        action_costs = compute_action_cost(df_valid_queries[features], df_all_cfs[features], features_to_vary)
        
        # Invalidation rates
        sg_series = pd.Series(cf_subgroups)
        air_xgb, sir_xgb = compute_invalidation_rates(inv_flags_xgb, sg_series)
        air_lr, sir_lr = compute_invalidation_rates(inv_flags_lr, sg_series)
        
        # Statistical Tests
        stat_tests_xgb = run_significance_tests(inv_flags_xgb, sg_series, group_a='thin-file', group_b='thick-file')
        stat_tests_lr = run_significance_tests(inv_flags_lr, sg_series, group_a='thin-file', group_b='thick-file')
        
        # Quality metrics
        qual = evaluate_counterfactual_quality(df_valid_queries, df_all_cfs, features_to_vary, features)
        
        recourse_summary = [
            {
                "model": "XGBoost",
                "air": air_xgb,
                "sir_thin_file": sir_xgb.get('thin-file', np.nan),
                "sir_thick_file": sir_xgb.get('thick-file', np.nan),
                "action_cost_mean": float(np.mean(action_costs)),
                "action_cost_std": float(np.std(action_costs)),
                "proximity": qual['proximity'],
                "sparsity": qual['sparsity'],
                "z_test_pval": stat_tests_xgb['z_test_pvalue'],
                "fisher_pval": stat_tests_xgb['fisher_exact_pvalue'],
                "cohens_h": stat_tests_xgb['cohens_h_effect_size'],
                "bootstrap_ci_95": str(stat_tests_xgb['bootstrap_ci_95'])
            },
            {
                "model": "LogisticRegression",
                "air": air_lr,
                "sir_thin_file": sir_lr.get('thin-file', np.nan),
                "sir_thick_file": sir_lr.get('thick-file', np.nan),
                "action_cost_mean": float(np.mean(action_costs)),
                "action_cost_std": float(np.std(action_costs)),
                "proximity": qual['proximity'],
                "sparsity": qual['sparsity'],
                "z_test_pval": stat_tests_lr['z_test_pvalue'],
                "fisher_pval": stat_tests_lr['fisher_exact_pvalue'],
                "cohens_h": stat_tests_lr['cohens_h_effect_size'],
                "bootstrap_ci_95": str(stat_tests_lr['bootstrap_ci_95'])
            }
        ]
        pd.DataFrame(recourse_summary).to_csv('results/Table_13_recourse_invalidation_and_quality.csv', index=False)
        print("  -> Exported results/Table_13_recourse_invalidation_and_quality.csv")
    else:
        df_all_cfs = pd.DataFrame()
        
    # -------------------------------------------------------------
    # 4. Sensitivity Suite Execution
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Executing Sensitivity & Ablation Suite...")
    run_comprehensive_sensitivity_suite(dataset_path=dataset_path, seeds=SEEDS[:3])

    # -------------------------------------------------------------
    # 5. Failure Case & Boundary-Jitter Analysis
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Profiling Boundary-Jitter Failure Cases...")
    if len(df_all_cfs) > 0:
        analyze_failure_cases_and_generate_waterfalls(
            df_t0=df_t0,
            df_t1=df_eval_last_seed,
            features=features,
            model_t0=xgb_t0_ref,
            model_t1=xgb_t1_ref,
            df_recourse=df_all_cfs,
            cf_subgroups=cf_subgroups,
            query_instances=df_valid_queries
        )

    # -------------------------------------------------------------
    # 6. Human Expert Feedback Artifacts
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Generating Interactive Human Feedback Artifacts...")
    generate_expert_feedback_artifacts()
    
    print("\n" + "="*80)
    print("SUCCESS: Full Pipeline Completed! All Tables & Figures Generated in results/")
    print("="*80)

if __name__ == '__main__':
    main()
