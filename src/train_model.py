"""
TeleXAI Model Training

Trains a LightGBM classifier to predict 5G tower failures within a 6-hour horizon.
Enforces strict time-based splitting to prevent temporal data leakage and uses 
class weighting to handle highly imbalanced failure events.
"""

import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import joblib

def time_based_split(df, target_col, exclude_cols, test_size=0.2):
    """Splits data chronologically to mimic real-world deployment."""
    # Ensure strict temporal ordering
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Isolate features (X) and target (y)
    X_train = train_df.drop(columns=exclude_cols + [target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=exclude_cols + [target_col])
    y_test = test_df[target_col]
    
    print(f"Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Testing period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return X_train, X_test, y_train, y_test, test_df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'processed', 'features.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    print(f"Loading engineered features from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    
    # Target variable and columns that must NOT be used as features
    target_col = 'label_fail_6h'
    exclude_cols = [
        'timestamp', 'tower_id', 'failure_event', 
        'root_cause', 'precursor_window'
    ]
    
    # 1. Time-based split
    print("\n--- Performing Time-Based Split ---")
    X_train, X_test, y_train, y_test, test_df_full = time_based_split(
        df, target_col, exclude_cols, test_size=0.2
    )
    
    # 2. Train LightGBM Model
    # 'balanced' class_weight forces the model to care about the rare minority class (failures)
    print("\n--- Training LightGBM Model ---")
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='aucpr', # Optimize for Precision-Recall Area Under Curve
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # 3. Evaluation
    print("\n--- Model Evaluation (Test Set) ---")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"PR-AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")
    
    # 4. Save Model and Test Data for XAI step
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'lgbm_baseline.joblib')
    joblib.dump(clf, model_path)
    print(f"\nSaved trained model to {model_path}")
    
    # Save the test set with true causes so we can evaluate XAI faithfulness later
    test_data_path = os.path.join(base_dir, 'data', 'processed', 'test_data.csv')
    test_df_full.to_csv(test_data_path, index=False)
    print(f"Saved test dataset (with ground truth) for XAI evaluation to {test_data_path}")

if __name__ == "__main__":
    main()