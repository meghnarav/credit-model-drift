import shap
import dice_ml
import pandas as pd
import numpy as np

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
    d = dice_ml.Data(dataframe=df_t0, continuous_features=features, outcome_name='RiskPerformance')
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method="random")
    return exp

def generate_recourse(exp, query_instances, features_to_vary, total_CFs=1):
    """
    Generates counterfactual recourse for denied applicants.
    Enforces the actionable feature constraints.
    """
    dice_cf = exp.generate_counterfactuals(
        query_instances,
        total_CFs=total_CFs,
        desired_class="opposite",
        features_to_vary=features_to_vary,
        permitted_range={'NetFractionRevolvingBurden': [0.0, 100.0]}
    )
    return dice_cf
