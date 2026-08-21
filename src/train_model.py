"""
TeleXAI Model Training & Selection
==================================

Trains LightGBM, XGBoost, and Random Forest classifiers using strict chronological 
splitting. Implements a "Train on Clean, Evaluate on Operational" paradigm to 
demonstrate high precision alerting in real-world NOC environments.
"""

import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix

def time_based_split(df, target_col, exclude_cols, test_size=0.2):
    """Splits chronologically, snapping the cutoff to avoid bisecting active precursor events."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    cutoff_time = df.iloc[split_idx]['timestamp']

    extend_to = []
    for tid, g in df.groupby('tower_id'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        is_p = g['precursor_window'].astype(bool)
        block_id = (is_p != is_p.shift()).cumsum()
        for _, block in g[is_p].groupby(block_id[is_p]):
            start, end = block['timestamp'].min(), block['timestamp'].max()
            if start < cutoff_time <= end:
                extend_to.append(end)
                
    if extend_to:
        cutoff_time = max(extend_to) + pd.Timedelta(hours=1)

    train_df = df[df['timestamp'] < cutoff_time].copy()
    test_df = df[df['timestamp'] >= cutoff_time].copy()

    X_train = train_df.drop(columns=exclude_cols + [target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=exclude_cols + [target_col])
    
    # y_test_clean is the denoised label for direct ML evaluation
    y_test_clean = test_df[target_col]

    # Reconstruct the strict "Operational Label" (any hour within 6h of failure, regardless of signal)
    # This represents the actual business requirement of the NOC engineers.
    test_df['operational_label'] = 0
    for tid, g in test_df.groupby('tower_id'):
        failures = g[g['failure_event'] == 1]
        for f_time in failures['timestamp']:
            mask = (g['timestamp'] >= f_time - pd.Timedelta(hours=6)) & (g['timestamp'] <= f_time)
            test_df.loc[g[mask].index, 'operational_label'] = 1
            
    y_test_operational = test_df['operational_label']

    print(f"Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Testing period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    return X_train, X_test, y_train, y_test_clean, y_test_operational, test_df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'processed', 'features.csv')
    models_dir = os.path.join(base_dir, 'models')

    print(f"Loading engineered features from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])

    target_col = 'label_fail_6h'
    exclude_cols = ['timestamp', 'tower_id', 'failure_event', 'root_cause', 'precursor_window']

    print("\n--- Performing Event-Aware Time-Based Split ---")
    X_train, X_test, y_train, y_test_clean, y_test_operational, test_df_full = time_based_split(
        df, target_col, exclude_cols
    )

    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nTraining Class Imbalance Ratio (Healthy : Failure) -> {imbalance_ratio:.1f} : 1")

    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=imbalance_ratio, random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    }

    os.makedirs(models_dir, exist_ok=True)
    print("\n--- Training & Evaluating Models ---")
    
    for name, model in models.items():
        print(f"\n{'='*40}\nTraining {name}...\n{'='*40}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print("\n[1] Evaluation on CLEAN Label (Data Science Metric)")
        pr_auc_clean = average_precision_score(y_test_clean, y_proba)
        print(f"PR-AUC: {pr_auc_clean:.4f}")
        
        print("\n[2] Evaluation on OPERATIONAL Label (Business Metric)")
        pr_auc_op = average_precision_score(y_test_operational, y_proba)
        print(f"PR-AUC: {pr_auc_op:.4f}")
        print(f"Confusion Matrix (Operational):\n{confusion_matrix(y_test_operational, y_pred)}")
        print(classification_report(y_test_operational, y_pred, digits=3))

        model_path = os.path.join(models_dir, f"{name.lower()}.joblib")
        joblib.dump(model, model_path)

    test_data_path = os.path.join(base_dir, 'data', 'processed', 'test_data.csv')
    test_df_full.to_csv(test_data_path, index=False)
    
    feature_cols_path = os.path.join(models_dir, 'feature_columns.joblib')
    joblib.dump(list(X_train.columns), feature_cols_path)
    print(f"\nSaved test dataset and feature column order successfully.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()