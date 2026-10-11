import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Publication-ready styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#1E1E1E'
plt.rcParams['axes.linewidth'] = 1.2

def save_dual_figure(fig, base_filename, output_dir='results/figures'):
    """
    Saves figure in both 300-DPI PNG and vector PDF formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path

def plot_roc_and_pr_curves(models_dict, df_eval, features, target='RiskPerformance', base_filename="Fig_01_roc_pr_curves"):
    """
    Generates ROC and Precision-Recall curves comparing LR and XGBoost across T0 and T1.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        'LR_T0': '#3B82F6', 'XGB_T0': '#1D4ED8',
        'LR_T1_unretrained': '#F59E0B', 'XGB_T1_unretrained': '#EF4444',
        'LR_T1_retrained': '#10B981', 'XGB_T1_retrained': '#047857'
    }
    
    y_true = df_eval[target].values
    X = df_eval[features]
    
    for label, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            
            c = colors.get(label, '#6B7280')
            ax_roc.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=c, lw=2)
            ax_pr.plot(recall, precision, label=f"{label} (AP = {pr_auc:.3f})", color=c, lw=2)
            
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax_roc.legend(loc="lower right", fontsize=9, frameon=True)
    ax_roc.grid(True, alpha=0.3)
    
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(loc="lower left", fontsize=9, frameon=True)
    ax_pr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_confusion_matrices(y_true, predictions_dict, base_filename="Fig_02_confusion_matrices"):
    """
    Plots normalized confusion matrices in a 2x3 grid for LR and XGBoost under T0, drifted, and retrained.
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for idx, (label, y_pred) in enumerate(predictions_dict.items()):
        if idx >= len(axes):
            break
        cm = confusion_matrix(y_true, y_pred, normalize='all') * 100
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=['Denied (0)', 'Approved (1)'],
                    yticklabels=['Denied (0)', 'Approved (1)'],
                    annot_kws={"size": 12, "weight": "bold"})
        axes[idx].set_title(f"{label}", fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("True Label", fontsize=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        
    plt.suptitle("Normalized Confusion Matrices (% of Population)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_calibration_curves(y_true, probas_dict, base_filename="Fig_03_calibration_curves"):
    """
    Plots probability calibration curves (reliability diagrams) with Brier scores.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", lw=1.5)
    
    colors = ['#2563EB', '#1D4ED8', '#D97706', '#DC2626', '#059669', '#047857']
    for idx, (label, y_prob) in enumerate(probas_dict.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)
        c = colors[idx % len(colors)]
        ax.plot(prob_pred, prob_true, "s-", label=f"{label} (Brier: {brier:.3f})", color=c, lw=1.8, markersize=5)
        
    ax.set_title("Probability Calibration Curves (Reliability Diagram)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_invalidation_vs_drift_sweep(sweep_df, base_filename="Fig_04_invalidation_vs_drift_sweep"):
    """
    Plots Aggregate and Subgroup Invalidation Rates vs drift delta with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=sweep_df, x="delta_pct", y="invalidation_rate", hue="subgroup",
        style="model", markers=True, dashes=False, lw=2.5, ax=ax, palette=['#DC2626', '#2563EB', '#059669']
    )
    
    ax.set_title("Recourse Invalidation Rate vs. Drift Magnitude (Δ)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Covariate Drift Perturbation Δ (%)", fontsize=11)
    ax.set_ylabel("Invalidation Rate (1.0 = 100% Broken Recourse)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title="Subgroup & Model", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_tornado_sensitivity_ranking(ranking_df, base_filename="Fig_05_tornado_sensitivity_ranking"):
    """
    Plots tornado chart ranking parameters by effect on Thin-vs-Thick invalidation gap.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = ranking_df.sort_values(by="effect_on_gap", ascending=True)
    
    y_pos = np.arange(len(df_sorted))
    colors = ['#EF4444' if x > 0 else '#3B82F6' for x in df_sorted['effect_on_gap']]
    
    ax.barh(y_pos, df_sorted['effect_on_gap'], color=colors, edgecolor='#1E1E1E', lw=1.2, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['parameter'], fontsize=10, fontweight='bold')
    ax.set_xlabel("Impact on Thin-vs-Thick Invalidation Disparity Gap (Δ Gap %)", fontsize=11)
    ax.set_title("Tornado Sensitivity Analysis: Determinants of Disparate Invalidation", fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', lw=1.2)
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(df_sorted['effect_on_gap']):
        ax.text(v + (0.5 if v >= 0 else -1.5), i, f"{v:+.1f}%", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_two_parameter_heatmap(heatmap_df, param_x, param_y, metric="invalidation_gap", base_filename="Fig_06_two_param_heatmap"):
    """
    Plots 2-parameter interaction heatmap (e.g. drift delta vs XGBoost depth).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    pivot = heatmap_df.pivot(index=param_y, columns=param_x, values=metric)
    
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': f"{metric} (%)"}, ax=ax, lw=1, linecolor='white')
    ax.set_title(f"Sensitivity Interaction: {param_x} vs {param_y}", fontsize=13, fontweight='bold')
    ax.set_xlabel(param_x, fontsize=11, fontweight='bold')
    ax.set_ylabel(param_y, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_shap_summary_comparison(shap_f0, shap_f1, feature_names, base_filename="Fig_07_shap_global_comparison"):
    """
    Plots paired global feature importance comparison bar chart for f0 vs f1.
    """
    mean_f0 = np.mean(np.abs(shap_f0), axis=0)
    mean_f1 = np.mean(np.abs(shap_f1), axis=0)
    
    # Top 10 features by f0
    top_idx = np.argsort(mean_f0)[-10:]
    top_features = [feature_names[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top_features))
    height = 0.35
    
    ax.barh(y + height/2, mean_f0[top_idx], height, label='f0 Baseline (T0)', color='#3B82F6', edgecolor='#1E1E1E')
    ax.barh(y - height/2, mean_f1[top_idx], height, label='f1 Retrained (T1)', color='#EF4444', edgecolor='#1E1E1E')
    
    ax.set_yticks(y)
    ax.set_yticklabels(top_features, fontsize=10, fontweight='bold')
    ax.set_xlabel("Mean |SHAP Value| (Attribution Magnitude)", fontsize=11)
    ax.set_title("Global SHAP Attribution Shift: Baseline (f0) vs Retrained (f1)", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)

def plot_failure_waterfall(feature_names, original_vals, shap_vals, base_value, title, base_filename):
    """
    Plots a custom waterfall attribution breakdown for a representative failure instance.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Select top 7 influential features for this instance
    top_idx = np.argsort(np.abs(shap_vals))[-7:]
    feats = [feature_names[i] for i in top_idx]
    vals = [original_vals[i] for i in top_idx]
    s_vals = [shap_vals[i] for i in top_idx]
    
    y = np.arange(len(feats))
    colors = ['#EF4444' if v < 0 else '#10B981' for v in s_vals]
    
    labels = [f"{feats[i]} = {vals[i]:.1f}" for i in range(len(feats))]
    ax.barh(y, s_vals, color=colors, edgecolor='#1E1E1E', height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.axvline(0, color='black', lw=1.2)
    ax.set_xlabel("SHAP Value Impact on Log-Odds", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return save_dual_figure(fig, base_filename)
