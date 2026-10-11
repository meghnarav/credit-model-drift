import pandas as pd
import numpy as np

def load_and_clean_data(file_path: str, imputation_strategy: str = 'median') -> pd.DataFrame:
    """
    Loads raw FICO HELOC dataset and sanitizes special negative codes.
    -9: No Bureau Record
    -8: Not applicable / structural zero
    -7: Condition not met (0 counts/durations)
    
    Imputation strategies for -9:
    - 'median': replace with column median (default)
    - 'zero': replace with 0
    - 'indicator': replace with 0 and add an indicator column `<col>_is_missing`
    """
    df = pd.read_csv(file_path)
    
    # Target encoding: 'Bad' -> 0 (Denied), 'Good' -> 1 (Approved)
    if 'RiskPerformance' in df.columns:
        df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
    
    special_cols = [c for c in df.columns if c != 'RiskPerformance']
    
    # Handle structural codes first
    for col in special_cols:
        # -8 -> structural zero
        df.loc[df[col] == -8, col] = 0
        # -7 -> condition not met
        df.loc[df[col] == -7, col] = 0
        
    if imputation_strategy == 'indicator':
        for col in special_cols:
            df[f'{col}_missing'] = (df[col] == -9).astype(int)
            df.loc[df[col] == -9, col] = 0
    elif imputation_strategy == 'zero':
        for col in special_cols:
            df.loc[df[col] == -9, col] = 0
    else:  # 'median'
        for col in special_cols:
            df.loc[df[col] == -9, col] = np.nan
        df.fillna(df.median(), inplace=True)
            
    return df

def assign_subgroups(df: pd.DataFrame, quantile_cutoff: float = 0.25) -> pd.DataFrame:
    """
    Stratifies applicants into 'thin-file', 'thick-file', or 'other'.
    Allows variable cutoff (e.g., 0.20, 0.25, 0.30) for sensitivity analysis.
    """
    p_low_msince = df['MSinceOldestTradeOpen'].quantile(quantile_cutoff)
    p_low_num = df['NumTotalTrades'].quantile(quantile_cutoff)
    
    p_high_msince = df['MSinceOldestTradeOpen'].quantile(0.50)
    p_high_num = df['NumTotalTrades'].quantile(0.50)
    
    def get_group(row):
        if row['MSinceOldestTradeOpen'] <= p_low_msince and row['NumTotalTrades'] <= p_low_num:
            return 'thin-file'
        elif row['MSinceOldestTradeOpen'] >= p_high_msince and row['NumTotalTrades'] >= p_high_num:
            return 'thick-file'
        return 'other'
        
    df['Subgroup'] = df.apply(get_group, axis=1)
    return df

def simulate_drift_split(
    df: pd.DataFrame, 
    drift_feature: str = 'NetFractionRevolvingBurden',
    delta: float = 0.25, 
    drift_type: str = 'covariate',
    seed: int = 42
):
    """
    Simulates dataset drift:
    - 'covariate': Biased subsampling on continuous feature (e.g. NetFractionRevolvingBurden or MSinceOldestTradeOpen or NumInqLast6M)
    - 'label': Prior shift altering bad-to-good outcome proportions in T1
    """
    rng = np.random.default_rng(seed)
    
    if drift_type == 'label':
        # Prior/label shift: decrease approval rate in T1
        good_mask = (df['RiskPerformance'] == 1)
        bad_mask = (df['RiskPerformance'] == 0)
        
        # T0: balanced split
        t0_good = df[good_mask].sample(frac=0.5, random_state=seed)
        t0_bad = df[bad_mask].sample(frac=0.5, random_state=seed)
        df_t0 = pd.concat([t0_good, t0_bad]).sample(frac=1.0, random_state=seed)
        
        # T1: drop good approvals by delta factor (e.g., severe credit freeze)
        rem_good = df[good_mask].drop(t0_good.index)
        rem_bad = df[bad_mask].drop(t0_bad.index)
        
        keep_good_frac = max(0.1, 0.8 - delta)
        t1_good = rem_good.sample(frac=keep_good_frac, random_state=seed)
        t1_bad = rem_bad.sample(frac=0.8, random_state=seed)
        df_t1 = pd.concat([t1_good, t1_bad]).sample(frac=1.0, random_state=seed)
        return df_t0, df_t1

    # Covariate shift on specified feature
    threshold = df[drift_feature].quantile(max(0.1, min(0.9, 0.5 + delta)))
    
    high_burden = df[df[drift_feature] >= threshold]
    low_burden = df[df[drift_feature] < threshold]
    
    # T0: baseline has more low burden / favorable values
    t0_low = low_burden.sample(frac=min(0.7, 0.5 + delta), random_state=seed)
    t0_high = high_burden.sample(frac=max(0.15, 0.4 - delta), random_state=seed)
    df_t0 = pd.concat([t0_low, t0_high]).sample(frac=1.0, random_state=seed)
    
    # T1: macroeconomic stress / shift
    t1_low = low_burden.drop(t0_low.index).sample(frac=max(0.2, 0.6 - delta), random_state=seed)
    t1_high = high_burden.drop(t0_high.index)
    df_t1 = pd.concat([t1_low, t1_high]).sample(frac=1.0, random_state=seed)
    
    return df_t0, df_t1


if __name__ == "__main__":
    # Create dummy data for testing purposes if raw dataset doesn't exist
    import os
    if not os.path.exists("../data/raw/heloc_dataset.csv"):
        print("Generating dummy HELOC dataset for testing...")
        os.makedirs("../data/raw", exist_ok=True)
        np.random.seed(42)
        n = 1000
        dummy_data = pd.DataFrame({
            'RiskPerformance': np.random.choice(['Bad', 'Good'], n),
            'MSinceOldestTradeOpen': np.random.randint(0, 500, n),
            'NumTotalTrades': np.random.randint(0, 50, n),
            'NetFractionRevolvingBurden': np.random.randint(0, 100, n),
            'NumSatisfactoryTrades': np.random.randint(0, 40, n),
            'ExternalRiskEstimate': np.random.randint(50, 95, n)
        })
        # Inject some special codes
        dummy_data.loc[0:50, 'MSinceOldestTradeOpen'] = -9
        dummy_data.loc[51:100, 'NumTotalTrades'] = -8
        dummy_data.to_csv("../data/raw/heloc_dataset.csv", index=False)
        print("Dummy data created at ../data/raw/heloc_dataset.csv")
