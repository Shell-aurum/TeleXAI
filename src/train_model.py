"""
TeleXAI Model Training & Selection
==================================
Trains LightGBM, XGBoost, and Random Forest classifiers using strict chronological
splitting. Evaluates models based on PR-AUC (Precision-Recall Area Under Curve)
to handle extreme class imbalance, and saves ALL models (not just the "best" one)
so they can each be explained with SHAP/LIME for comparison.
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
    """
    Splits chronologically, but snaps the cutoff so no single failure
    event's precursor window is bisected across train and test. A naive
    row-count cutoff can split one ramp (e.g. hours 1-24 of a 36h event
    in train, hours 25-36 in test), which leaks a partially-seen event
    into evaluation and inflates apparent recall/precision.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    cutoff_time = df.iloc[split_idx]['timestamp']

    # Only extend past precursor blocks that are ACTIVE at the naive
    # cutoff (start before it, end at/after it) - not any other
    # precursor activity nearby.
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

    train_df = df[df['timestamp'] < cutoff_time]
    test_df = df[df['timestamp'] >= cutoff_time]

    X_train = train_df.drop(columns=exclude_cols + [target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=exclude_cols + [target_col])
    y_test = test_df[target_col]

    print(f"Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Testing period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"y_train positive rate: {y_train.mean():.4f} ({y_train.sum()} / {len(y_train)})")
    print(f"y_test positive rate:  {y_test.mean():.4f} ({y_test.sum()} / {len(y_test)})")

    return X_train, X_test, y_train, y_test, test_df


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'processed', 'features.csv')
    models_dir = os.path.join(base_dir, 'models')

    print(f"Loading engineered features from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])

    target_col = 'label_fail_6h'
    exclude_cols = ['timestamp', 'tower_id', 'failure_event', 'root_cause', 'precursor_window']

    # 1. Time-based split (event-aware)
    print("\n--- Performing Event-Aware Time-Based Split ---")
    X_train, X_test, y_train, y_test, test_df_full = time_based_split(df, target_col, exclude_cols)

    # Calculate exact imbalance ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / pos_count
    print(f"\nTraining Class Imbalance Ratio (Healthy : Failure) -> {imbalance_ratio:.1f} : 1")

    # 2. Define Models
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

    # 3. Train and Compare
    results = {}
    best_pr_auc = 0
    best_model_name = None

    print("\n--- Training & Evaluating Models ---")
    os.makedirs(models_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_proba)
        results[name] = pr_auc

        print(f"{name} PR-AUC: {pr_auc:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(classification_report(y_test, y_pred, digits=3))

        # Save EVERY model, not just the winner - each one gets explained
        # with SHAP/LIME separately, so all three need to survive this step.
        model_path = os.path.join(models_dir, f"{name.lower()}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name

    print("\n--- Summary ---")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        marker = " <-- best PR-AUC" if name == best_model_name else ""
        print(f"{name}: {score:.4f}{marker}")
    print(
        f"\nNote: 'best PR-AUC' is measured on the same held-out test set "
        f"used for reporting. With only ~{y_test.sum()} positive test rows, "
        f"small PR-AUC differences between models are not necessarily "
        f"meaningful - treat this as a rough guide, not a definitive ranking."
    )

    # 4. Save test data (shared across all models for XAI evaluation)
    test_data_path = os.path.join(base_dir, 'data', 'processed', 'test_data.csv')
    test_df_full.to_csv(test_data_path, index=False)
    print(f"\nSaved test dataset to {test_data_path}")

    # Also save X_train/X_test column order - SHAP/LIME need this later
    # and it's easy to silently get wrong if features.csv columns change.
    feature_cols_path = os.path.join(models_dir, 'feature_columns.joblib')
    joblib.dump(list(X_train.columns), feature_cols_path)
    print(f"Saved feature column order to {feature_cols_path}")


if __name__ == "__main__":
    main()