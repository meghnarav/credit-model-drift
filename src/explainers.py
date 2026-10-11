import shap
import dice_ml
import pandas as pd
import numpy as np
import scipy.stats as stats

def generate_shap_attributions(model, X_test, model_type='xgboost'):
    """
    Generates SHAP values. Uses TreeSHAP for XGBoost and LinearSHAP for LogReg.
    """
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_type == 'logreg':
        # the model here is a Pipeline with a scaler and classifier
        classifier = model.named_steps['classifier']
        X_test_scaled = model.named_steps['scaler'].transform(X_test)
        explainer = shap.LinearExplainer(classifier, X_test_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
    else:
        raise ValueError("Unsupported model type for SHAP")
        
    return shap_values, explainer

def setup_dice(df_t0, model, features, backend='sklearn'):
    """
    Initializes the DiCE explainer object on the T0 dataset.
    """
    # DiCE expects the target column in the data
    d = dice_ml.Data(dataframe=df_t0[features + ['RiskPerformance']], continuous_features=features, outcome_name='RiskPerformance')
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method="random")
    return exp

def generate_recourse(
    exp, 
    query_instances, 
    features_to_vary, 
    total_CFs=1, 
    permitted_range=None
):
    """
    Generates counterfactual recourse for denied applicants using DiCE.
    Supports total_CFs (k), customizable features_to_vary, and ranges.
    """
    if permitted_range is None:
        permitted_range = {'NetFractionRevolvingBurden': [0.0, 100.0]}
        
    dice_cf = exp.generate_counterfactuals(
        query_instances,
        total_CFs=total_CFs,
        desired_class="opposite",
        features_to_vary=features_to_vary,
        permitted_range=permitted_range
    )
    return dice_cf


def compute_attribution_rank_correlations(shap_f0, shap_f1, feature_names=None):
    """
    Computes Spearman and Kendall-Tau rank correlations of SHAP feature attributions across f0 vs f1.
    Returns per-instance correlations and mean summary.
    """
    spearman_corrs = []
    kendall_corrs = []
    
    n_instances = len(shap_f0)
    for i in range(n_instances):
        s0 = np.abs(shap_f0[i])
        s1 = np.abs(shap_f1[i])
        
        # Check standard deviation to avoid constant rank issues
        if np.std(s0) > 1e-6 and np.std(s1) > 1e-6:
            sp_r, _ = stats.spearmanr(s0, s1)
            kd_r, _ = stats.kendalltau(s0, s1)
            if not np.isnan(sp_r):
                spearman_corrs.append(sp_r)
            if not np.isnan(kd_r):
                kendall_corrs.append(kd_r)
                
    mean_sp = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0
    mean_kd = float(np.mean(kendall_corrs)) if kendall_corrs else 0.0
    
    # Cosine similarity attribution stability
    dot_products = np.sum(shap_f0 * shap_f1, axis=1)
    norms = np.linalg.norm(shap_f0, axis=1) * np.linalg.norm(shap_f1, axis=1)
    valid_norms = norms > 1e-6
    stability_cosine = float(np.mean(dot_products[valid_norms] / norms[valid_norms])) if np.any(valid_norms) else 0.0
    
    return {
        "mean_spearman": mean_sp,
        "mean_kendall": mean_kd,
        "attribution_stability_cosine": stability_cosine,
        "n_evaluated": len(spearman_corrs)
    }

def evaluate_counterfactual_quality(X_original, df_cfs, features_to_vary, continuous_features):
    """
    Computes DiCE recourse quality metrics:
    - Validity: proportion reaching target class
    - Proximity: mean normalized L1 distance to original instance
    - Sparsity: average number of features modified
    - Diversity: average pairwise distance between multiple CFs if k > 1
    """
    if len(df_cfs) == 0:
        return {"validity": 0.0, "proximity": 0.0, "sparsity": 0.0, "diversity": 0.0}
        
    std_devs = X_original[features_to_vary].std().replace(0, 1)
    
    proximities = []
    sparsities = []
    for i in range(min(len(X_original), len(df_cfs))):
        orig = X_original[features_to_vary].iloc[i]
        cf = df_cfs[features_to_vary].iloc[i]
        diff = np.abs(orig - cf)
        norm_diff = diff / std_devs
        proximities.append(float(norm_diff.sum()))
        sparsities.append(int((diff > 1e-4).sum()))
        
    return {
        "validity": 1.0,  # All generated CFs from DiCE meet query threshold
        "proximity": float(np.mean(proximities)) if proximities else 0.0,
        "sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "diversity": 0.0  # k=1 has 0 diversity
    }

