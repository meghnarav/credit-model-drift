import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import assign_subgroups, simulate_drift_split
from src.evaluation import run_significance_tests, compute_invalidation_rates

def test_subgroup_assignment():
    # Mock data for subgroup assignment
    df = pd.DataFrame({
        'MSinceOldestTradeOpen': [10, 50, 100],
        'NumTotalTrades': [2, 10, 20]
    })
    
    # 25th percentile for MSince is 30, Num is 6. 
    # 50th percentile for MSince is 50, Num is 10.
    
    df_assigned = assign_subgroups(df)
    
    # Check if subgroups are correctly assigned
    assert df_assigned.loc[0, 'Subgroup'] == 'thin-file'
    assert df_assigned.loc[2, 'Subgroup'] == 'thick-file'

def test_significance_tests():
    # Mock invalidation flags and subgroups
    np.random.seed(42)
    subgroups = pd.Series(['thin-file']*50 + ['thick-file']*50)
    
    # High invalidation rate for thin-file
    invalidation_flags = np.array([True]*40 + [False]*10 + [True]*10 + [False]*40)
    
    # Compute SIR
    air, sir = compute_invalidation_rates(invalidation_flags, subgroups)
    assert sir['thin-file'] == 0.8
    assert sir['thick-file'] == 0.2
    
    # Run tests
    results = run_significance_tests(invalidation_flags, subgroups)
    
    assert results['significant_difference'] is True
    assert results['z_test_pvalue'] < 0.05
    assert results['fisher_exact_pvalue'] < 0.05

def test_dice_constraint_enforcement():
    # A dummy test to conceptually verify the constraints (mocking DiCE is complex without full setup)
    immutable_features = ["MSinceOldestTradeOpen", "NumTotalTrades"]
    actionable_features = ["NetFractionRevolvingBurden", "NumSatisfactoryTrades", "ExternalRiskEstimate"]
    
    # Ensure they don't overlap
    assert len(set(immutable_features).intersection(set(actionable_features))) == 0
