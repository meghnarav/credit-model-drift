import streamlit as st
import pandas as pd
import numpy as np

# Apply Bento UI Styling
st.set_page_config(page_title="Recourse Robustness Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Bento Grid & Dark Mode Financial Styling
st.markdown("""
<style>
    .bento-card {
        background-color: #1e1e2e;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #2d2d3f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        color: #e0e0e0;
    }
    .bento-card h3 {
        color: #89b4fa;
        margin-top: 0;
        font-size: 1.25rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #cba6f7;
    }
    .badge-approved {
        background-color: #a6e3a1;
        color: #11111b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-denied {
        background-color: #f38ba8;
        color: #11111b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-risk {
        background-color: #fab387;
        color: #11111b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .recourse-step {
        background-color: #313244;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #89b4fa;
    }
    .immutable-step {
        border-left: 4px solid #f38ba8;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

st.title("Credit Model Drift & Recourse Verification")
st.markdown("Evaluating Subgroup-Differentiated Counterfactual Invalidation")

# Top row: Core metrics & Reliability
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="bento-card">
        <h3>Card 1: Applicant Profile</h3>
        <p><strong>Subgroup:</strong> Thin-file</p>
        <p><strong>Base Prediction (f₀):</strong> <span class="badge-denied">Denied</span></p>
        <p><strong>Confidence Score:</strong> 0.23 (Threshold: 0.50)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="bento-card">
        <h3>Card 4: Reliability Indicator</h3>
        <p>Empirical Invalidation Risk under Macroeconomic Drift</p>
        <div class="metric-value">42.5%</div>
        <p><span class="badge-risk">High Vulnerability</span> for Thin-file applicants</p>
    </div>
    """, unsafe_allow_html=True)

# Bottom row: XAI and Recourse
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("""
    <div class="bento-card">
        <h3>Card 2: XAI Panel (SHAP Attributions)</h3>
        <p><em>Placeholder for SHAP Waterfall Chart</em></p>
        <div style="height: 200px; background-color: #313244; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
            SHAP Waterfall Visualization
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="bento-card">
        <h3>Card 3: Recourse Recommendations</h3>
        <p>Path to Approval (f₀(x*) = 1)</p>
        
        <h4>Actionable Adjustments</h4>
        <div class="recourse-step">
            ↓ Reduce <strong>NetFractionRevolvingBurden</strong> from 85.0 to 45.2
        </div>
        <div class="recourse-step">
            ↑ Increase <strong>ExternalRiskEstimate</strong> from 62 to 68
        </div>
        
        <h4 style="margin-top: 16px;">Immutable Context</h4>
        <div class="recourse-step immutable-step">
            • <strong>MSinceOldestTradeOpen</strong>: 24 months (Locked)
        </div>
        <div class="recourse-step immutable-step">
            • <strong>NumTotalTrades</strong>: 5 (Locked)
        </div>
    </div>
    """, unsafe_allow_html=True)
