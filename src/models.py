import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_baseline_models(df_t0, features, target='RiskPerformance'):
    """
    Trains baseline Logistic Regression and XGBoost classifiers on T0.
    """
    X_train = df_t0[features]
    y_train = df_t0[target]
    
    # Logistic Regression pipeline
    lr_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return lr_model, xgb_model

def train_retrained_models(df_t1, features, target='RiskPerformance'):
    """
    Trains retrained models on T1 distribution (after drift).
    """
    return train_baseline_models(df_t1, features, target)

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
