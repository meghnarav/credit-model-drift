import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_baseline_models(
    df_t0, 
    features, 
    target='RiskPerformance',
    seed=42,
    lr_c=1.0,
    xgb_max_depth=4,
    xgb_learning_rate=0.1,
    xgb_n_estimators=100
):
    """
    Trains baseline Logistic Regression and XGBoost classifiers on T0 with configurable seed & hyperparameters.
    """
    X_train = df_t0[features]
    y_train = df_t0[target]
    
    # Logistic Regression pipeline
    lr_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=lr_c, random_state=seed, max_iter=1000))
    ])
    lr_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=seed,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        n_estimators=xgb_n_estimators
    )
    xgb_model.fit(X_train, y_train)
    
    return lr_model, xgb_model

def train_retrained_models(
    df_t1, 
    features, 
    target='RiskPerformance',
    seed=42,
    lr_c=1.0,
    xgb_max_depth=4,
    xgb_learning_rate=0.1,
    xgb_n_estimators=100
):
    """
    Trains retrained models on T1 distribution (after drift).
    """
    return train_baseline_models(
        df_t1, 
        features, 
        target=target,
        seed=seed,
        lr_c=lr_c,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_n_estimators=xgb_n_estimators
    )


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
