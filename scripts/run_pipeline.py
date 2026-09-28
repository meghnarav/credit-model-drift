import os
import sys
import pandas as pd
import json

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_and_clean_data, assign_subgroups, simulate_drift_split
from src.models import train_baseline_models, train_retrained_models, save_model
from src.explainers import generate_shap_attributions, setup_dice, generate_recourse
from src.evaluation import evaluate_recourse_invalidation, compute_invalidation_rates, run_significance_tests, output_results

def main():
    print("Loading data...")
    dataset_path = 'data/raw/heloc_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please download the FICO HELOC dataset and place it in the data/raw/ directory.")
        return

    df = load_and_clean_data(dataset_path)
    df = assign_subgroups(df)
    
    print("Simulating drift...")
    df_t0, df_t1 = simulate_drift_split(df)
    
    features = [c for c in df.columns if c not in ['RiskPerformance', 'Subgroup']]
    
    print("Training baseline models (T0)...")
    lr_t0, xgb_t0 = train_baseline_models(df_t0, features)
    print("Training retrained models (T1)...")
    lr_t1, xgb_t1 = train_retrained_models(df_t1, features)
    
    save_model(xgb_t0, 'results/models/xgb_t0.pkl')
    save_model(xgb_t1, 'results/models/xgb_t1.pkl')
    
    print("Generating SHAP attributions on T0 test sample...")
    # Use a small sample to speed up demo
    X_sample = df_t0[features].sample(min(100, len(df_t0)), random_state=42)
    shap_vals, explainer = generate_shap_attributions(xgb_t0, X_sample, model_type='xgboost')
    
    print("Setting up DiCE Counterfactual Explainer...")
    dice_exp = setup_dice(df_t0, xgb_t0, features)
    
    # We want to generate recourse for denied applicants
    predictions_t0 = xgb_t0.predict(df_t0[features])
    denied = df_t0[predictions_t0 == 0]
    
    if len(denied) > 0:
        # Sample 20 denied applicants to generate recourse for
        query_instances = denied[features].sample(min(20, len(denied)), random_state=42)
        print(f"Generating recourse for {len(query_instances)} denied applicants...")
        
        # Enforce DiCE constraints
        features_to_vary = ["NetFractionRevolvingBurden", "NumSatisfactoryTrades", "ExternalRiskEstimate"]
        
        # Generate counterfactuals
        dice_cf = generate_recourse(dice_exp, query_instances, features_to_vary, total_CFs=1)
        
        # Extract counterfactuals
        all_cfs = []
        cf_subgroups = []
        for i, exp in enumerate(dice_cf.cf_examples_list):
            if exp.final_cfs_df is not None:
                all_cfs.append(exp.final_cfs_df)
                # Track the subgroup for each generated CF
                query_idx = query_instances.index[i]
                subgroup = denied.loc[query_idx, 'Subgroup']
                cf_subgroups.extend([subgroup] * len(exp.final_cfs_df))
        
        if all_cfs:
            df_recourse = pd.concat(all_cfs)
            
            print("Evaluating recourse invalidation under drift (T1)...")
            invalidation_flags = evaluate_recourse_invalidation(xgb_t1, df_recourse, features)
            
            subgroups_series = pd.Series(cf_subgroups)
            
            print("Computing Aggregate and Subgroup Invalidation Rates (AIR, SIR)...")
            air, sir = compute_invalidation_rates(invalidation_flags, subgroups_series)
            
            print("Running significance tests...")
            tests = run_significance_tests(invalidation_flags, subgroups_series)
            
            os.makedirs('results', exist_ok=True)
            output_results(air, sir, tests, output_dir='results')
            
            print("\n==================================")
            print("Pipeline Complete! Results saved.")
            print(f"AIR: {air:.2%}")
            print(f"SIR (Thin-file): {sir.get('thin-file', float('nan')):.2%}")
            print(f"SIR (Thick-file): {sir.get('thick-file', float('nan')):.2%}")
            print("==================================")
        else:
            print("No counterfactuals found.")
    else:
        print("No denied applicants found to generate recourse for.")

if __name__ == '__main__':
    main()
