import pandas as pd
import numpy as np

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw FICO HELOC dataset and sanitizes special negative codes.
    -9: Imputed with NaN initially, then handled by model or median imputation.
    -8: Treated as structural zero (mapped to 0).
    -7: Mapped to 0 for counts/durations.
    """
    df = pd.read_csv(file_path)
    
    # Target encoding: 'Bad' -> 0 (Denied), 'Good' -> 1 (Approved)
    if 'RiskPerformance' in df.columns:
        df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
    
    # Handle special codes
    for col in df.columns:
        if col != 'RiskPerformance':
            # -8 -> 0 (Not applicable -> structural zero)
            df.loc[df[col] == -8, col] = 0
            # -7 -> 0 (Condition not met -> 0 counts/durations)
            df.loc[df[col] == -7, col] = 0
            # -9 -> NaN (No Bureau Record) - for simplicity here, we'll impute with median later
            df.loc[df[col] == -9, col] = np.nan
            
    # Simple median imputation for -9s to allow standard classifiers to run
    df.fillna(df.median(), inplace=True)
            
    return df

def assign_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratifies applicants into 'thin-file', 'thick-file', or 'other'.
    """
    p25_msince = df['MSinceOldestTradeOpen'].quantile(0.25)
    p25_num = df['NumTotalTrades'].quantile(0.25)
    
    p50_msince = df['MSinceOldestTradeOpen'].quantile(0.50)
    p50_num = df['NumTotalTrades'].quantile(0.50)
    
    def get_group(row):
        if row['MSinceOldestTradeOpen'] <= p25_msince and row['NumTotalTrades'] <= p25_num:
            return 'thin-file'
        elif row['MSinceOldestTradeOpen'] >= p50_msince and row['NumTotalTrades'] >= p50_num:
            return 'thick-file'
        return 'other'
        
    df['Subgroup'] = df.apply(get_group, axis=1)
    return df

def simulate_drift_split(df: pd.DataFrame, target_col='RiskPerformance'):
    """
    Simulates covariate drift by creating a baseline (T0) and post-drift (T1) distribution.
    Biased subsampling on NetFractionRevolvingBurden to simulate macroeconomic shift.
    T1 will have a higher proportion of applicants with higher burden.
    """
    # Define a threshold for high burden (e.g., 75th percentile)
    threshold = df['NetFractionRevolvingBurden'].quantile(0.75)
    
    high_burden = df[df['NetFractionRevolvingBurden'] >= threshold]
    low_burden = df[df['NetFractionRevolvingBurden'] < threshold]
    
    # T0: more low burden (baseline)
    t0_low = low_burden.sample(frac=0.6, random_state=42)
    t0_high = high_burden.sample(frac=0.3, random_state=42)
    df_t0 = pd.concat([t0_low, t0_high]).sample(frac=1, random_state=42)
    
    # T1: more high burden (macroeconomic stress)
    t1_low = low_burden.drop(t0_low.index).sample(frac=0.5, random_state=42)
    t1_high = high_burden.drop(t0_high.index)
    df_t1 = pd.concat([t1_low, t1_high]).sample(frac=1, random_state=42)
    
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
