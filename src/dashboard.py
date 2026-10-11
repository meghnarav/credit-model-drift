import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Credit Model Drift & Recourse Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Neo-Brutalist & Clean Financial Dark/Light Aesthetic)
st.markdown("""
<style>
    .main { background-color: #FAF7F0; }
    .stApp { background-color: #FAF7F0; color: #1E1E1E; }
    .metric-card {
        background-color: #FFFDF9;
        border: 3px solid #1E1E1E;
        box-shadow: 5px 5px 0px #1E1E1E;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .badge-thin {
        background-color: #FDE047;
        color: #1E1E1E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        border: 2px solid #1E1E1E;
    }
    .badge-danger {
        background-color: #F87171;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        border: 2px solid #1E1E1E;
    }
    .badge-success {
        background-color: #4ADE80;
        color: #1E1E1E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        border: 2px solid #1E1E1E;
    }
    .recourse-box {
        background-color: #FEF3C7;
        border: 2px solid #1E1E1E;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Credit Risk Prediction Under Model Drift")
st.caption("Evaluating Robustness, Subgroup Fairness, and Algorithmic Recourse Fragility on FICO HELOC")

# Sidebar navigation
menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "Interactive Audit Dashboard",
        "Base Model Performance (Table 01 & 02)",
        "Sensitivity & Ablation Curves",
        "Failure Case Profiler",
        "Human Underwriter Review (IML)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Research Highlights:**
- **Thin-File Recall Collapse:** $59.8\\% \\to 28.1\\%$ under drift.
- **Recourse Invalidation:** Up to $50.0\\%$ of approved recourse plans fail post-retraining.
- **Statistical Significance:** Cohen's $h$ & Bootstrap 95% CIs evaluated across 5 random seeds.
""")

if menu == "Interactive Audit Dashboard":
    st.subheader("Auditor Bento Grid: Live Applicant Decision & Recourse")
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1. Applicant Profile</h3>
            <p><strong>Applicant ID:</strong> <code>APP-9824</code></p>
            <p><strong>Segment:</strong> <span class="badge-thin">Thin-File</span></p>
            <hr style="border: 1px solid #1E1E1E; margin: 10px 0;">
            <p><strong>External Risk Estimate:</strong> 62</p>
            <p><strong>Months in File:</strong> 14 (Locked)</p>
            <p><strong>Total Trade Lines:</strong> 5 (Locked)</p>
            <p><strong>Revolving Debt Burden:</strong> 85.0%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>2. Model Decision & Invalidation Risk</h3>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="color: #DC2626; margin: 0; font-size: 2.8rem; font-weight: 900;">DENIED</h1>
                    <span class="badge-danger">f₀(x) = 0</span>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-weight: bold;">Approval Probability</p>
                    <h2 style="margin: 0;">23.4%</h2>
                </div>
            </div>
            <hr style="border: 1px solid #1E1E1E; margin: 15px 0;">
            <p><strong>Recourse Invalidation Vulnerability:</strong></p>
            <div style="background-color: #E5E7EB; border: 2px solid #1E1E1E; border-radius: 8px; height: 24px; overflow: hidden;">
                <div style="background-color: #EF4444; width: 66.7%; height: 100%;"></div>
            </div>
            <p style="font-size: 0.85rem; color: #B91C1C; font-weight: 700; margin-top: 6px;">
                ⚠️ 66.7% Thin-File Alert: Prescribed recourse has high likelihood of breaking upon model retraining.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>3. DiCE Action Plan</h3>
            <p style="font-size: 0.85rem; font-weight: 600;">Prescribed actions to achieve approval under f₀:</p>
            <div class="recourse-box">
                ↓ <strong>Revolving Burden:</strong> 85% → 45%
            </div>
            <div class="recourse-box">
                ↑ <strong>Risk Estimate:</strong> 62 → 68
            </div>
            <hr style="border: 1px solid #1E1E1E; margin: 10px 0;">
            <p style="font-size: 0.85rem;"><strong>Normalized L₁ Action Cost:</strong> <code>5.62</code></p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Global & Local SHAP Attributions")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        if os.path.exists("results/figures/Fig_07_shap_global_comparison.png"):
            st.image("results/figures/Fig_07_shap_global_comparison.png", caption="Fig 07: Global SHAP Feature Importance Shift (f₀ vs f₁)")
    with fig_col2:
        if os.path.exists("results/figures/Fig_08_waterfall_thin-file_case1.png"):
            st.image("results/figures/Fig_08_waterfall_thin-file_case1.png", caption="Fig 08: Representative Thin-File Decision Boundary Waterfall")

elif menu == "Base Model Performance (Table 01 & 02)":
    st.subheader("Stage 1 Base Model Benchmark Across 5 Random Seeds")
    if os.path.exists("results/Table_01_baseline_model_performance.csv"):
        df_tab1 = pd.read_csv("results/Table_01_baseline_model_performance.csv")
        st.dataframe(df_tab1, use_container_width=True)
    
    st.subheader("Subgroup Fairness Breakdown: Thin-File vs Thick-File")
    if os.path.exists("results/Table_02_subgroup_fairness_breakdown.csv"):
        df_tab2 = pd.read_csv("results/Table_02_subgroup_fairness_breakdown.csv")
        st.dataframe(df_tab2, use_container_width=True)
        
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("results/figures/Fig_01_roc_pr_curves.png"):
            st.image("results/figures/Fig_01_roc_pr_curves.png", caption="Fig 01: ROC and Precision-Recall Curves")
    with col_b:
        if os.path.exists("results/figures/Fig_02_confusion_matrices.png"):
            st.image("results/figures/Fig_02_confusion_matrices.png", caption="Fig 02: Normalized Confusion Matrix Grid")

elif menu == "Sensitivity & Ablation Curves":
    st.subheader("Multi-Parameter Drift & Invalidation Sensitivity")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if os.path.exists("results/figures/Fig_04_invalidation_vs_drift_sweep.png"):
            st.image("results/figures/Fig_04_invalidation_vs_drift_sweep.png", caption="Fig 04: Invalidation Rate vs Drift Magnitude (Δ)")
    with col_s2:
        if os.path.exists("results/figures/Fig_05_tornado_sensitivity_ranking.png"):
            st.image("results/figures/Fig_05_tornado_sensitivity_ranking.png", caption="Fig 05: Tornado Ranking of Drivers of the Invalidation Gap")
            
    if os.path.exists("results/figures/Fig_06_two_param_heatmap.png"):
        st.image("results/figures/Fig_06_two_param_heatmap.png", caption="Fig 06: Interaction Heatmap: Learning Rate vs Tree Depth")

elif menu == "Failure Case Profiler":
    st.subheader("Decision-Boundary Jitter & Recourse Invalidation Failures")
    if os.path.exists("results/Table_10_failure_patterns_by_feature.csv"):
        df_pats = pd.read_csv("results/Table_10_failure_patterns_by_feature.csv")
        st.dataframe(df_pats, use_container_width=True)
        
    st.subheader("Failure Waterfall Gallery")
    w_cols = st.columns(2)
    waterfalls = [
        "results/figures/Fig_08_waterfall_thin-file_case1.png",
        "results/figures/Fig_08_waterfall_thin-file_case2.png",
        "results/figures/Fig_08_waterfall_thick-file_case1.png",
        "results/figures/Fig_08_waterfall_other_case1.png"
    ]
    for idx, w_path in enumerate(waterfalls):
        if os.path.exists(w_path):
            with w_cols[idx % 2]:
                st.image(w_path, caption=os.path.basename(w_path))

elif menu == "Human Underwriter Review (IML)":
    st.subheader("Human-in-the-Loop Credit Committee Review")
    st.markdown("""
    Underwriters audit algorithmic recourse advice to assess practical actionability and economic plausibility.
    """)
    if os.path.exists("results/figures/Fig_09_expert_feedback_summary.png"):
        st.image("results/figures/Fig_09_expert_feedback_summary.png", caption="Fig 09: Human Expert Scores and Committee Verdict Distribution")
        
    st.subheader("Submit Underwriter Recourse Audit")
    with st.form("underwriter_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            app_id = st.text_input("Applicant ID", "APP-9824")
            rev_id = st.selectbox("Auditor Persona", ["REV-001 (Senior Underwriter)", "REV-002 (Fair Lending Officer)", "REV-003 (Risk Modeler)"])
            subgroup = st.selectbox("Subgroup", ["thin-file", "thick-file", "other"])
        with col_f2:
            act_score = st.slider("Actionability Score (1=Impossible, 5=Easily Actionable)", 1, 5, 2)
            plaus_score = st.slider("Economic Plausibility (1=Implausible, 5=Highly Realistic)", 1, 5, 2)
            verdict = st.selectbox("Final Committee Verdict", ["REJECT", "MODIFY", "ACCEPT"])
            
        notes = st.text_area("Audit Override Notes", "Revolving debt reduction target requires unrealistic income surplus.")
        submitted = st.form_submit_button("Submit Audit Decision")
        
        if submitted:
            csv_path = "results/expert_feedback.csv"
            new_row = pd.DataFrame([{
                'application_id': app_id,
                'reviewer_id': rev_id,
                'applicant_subgroup': subgroup,
                'actionability_score': act_score,
                'plausibility_score': plaus_score,
                'verdict': verdict,
                'reviewer_notes': notes
            }])
            if os.path.exists(csv_path):
                new_row.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                new_row.to_csv(csv_path, index=False)
            st.success("✅ Audit recorded and appended to results/expert_feedback.csv")
