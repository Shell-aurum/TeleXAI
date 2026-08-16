"""
TeleXAI Feature Engineering:

Transforms raw hourly telemetry into a rich feature set for predictive modeling.
Calculates rolling statistics and rate of change metrics while strictly 
grouping by tower_id to prevent data leakage between distinct cell sites.
"""

import pandas as pd
import numpy as np
import os

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df.timestamp.dt.dayofweek

    df['hour_sin'] = np.sin(df.hour *(2. * np.pi / 24))
    df['hour_cos'] = np.cos(df.hour *(2. * np.pi / 24))
    df['dow_sin'] = np.sin(df.dayofweek *(2. * np.pi / 7))
    df['dow_cos'] = np.cos(df.dayofweek *(2. * np.pi / 7))

    return df.drop(columns=['hour', 'dayofweek'])

def create_lag_features(df: pd.DataFrame, feature_cols: list, lags=[1, 2]) -> pd.DataFrame:
    grouped = df.groupby('tower_id')[feature_cols]
    for lag in lags:
        shifted = grouped.shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in feature_cols]
        df = df.join(shifted)
    return df

def create_rolling_and_deviation_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    grouped = df.groupby('tower_id')[feature_cols]
    for window in [3, 6]:
        rolling_mean = grouped.rolling(window).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f'{col}_roll_mean_{window}h' for col in feature_cols]
        df = df.join(rolling_mean)
        
        rolling_std = grouped.rolling(window).std().reset_index(level=0, drop=True)
        rolling_std.columns = [f'{col}_roll_std_{window}h' for col in feature_cols]
        df = df.join(rolling_std)
        
        for col in feature_cols:
            df[f'{col}_dev_{window}h'] = df[col] - df[f'{col}_roll_mean_{window}h']
            
    diff_1h = grouped.diff(1)
    diff_1h.columns = [f'{col}_diff_1h' for col in feature_cols]
    df = df.join(diff_1h)
    
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'raw', 'telemetry.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    output_csv = os.path.join(output_dir, 'features.csv')

    print(f"Loading raw telemetry data from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    df = df.sort_values(by=['tower_id', 'timestamp']).reset_index(drop=True)
    
    if 'urban' in df.columns:
        df['urban'] = df['urban'].astype(int)
    
    telemetry_cols = [
        'sinr_db', 'rsrp_dbm', 'packet_loss_pct', 'jitter_ms',
        'latency_ms', 'prb_utilization_pct', 'connected_users', 
        'hardware_temp_c', 'throughput_mbps'
    ]
    
    print("Engineering cyclical time features...")
    df = create_time_features(df)
    
    print("Engineering exact lag features...")
    df = create_lag_features(df, telemetry_cols, lags=[1, 2])
    
    print("Engineering rolling, deviation, and rate-of-change features...")
    df = create_rolling_and_deviation_features(df, telemetry_cols)
    
    # root_cause / precursor_window are ground-truth METADATA, not features.
    # They are NaN/False for the ~99% of rows that are healthy - that's
    # expected, not missing data. Only drop rows where the WINDOW-derived
    # features (lag_*, roll_*, dev_*, diff_*) themselves are NaN, which
    # happens for the first few hours of each tower's history.
    metadata_cols = ['failure_event', 'root_cause', 'precursor_window']
    window_feature_cols = [c for c in df.columns if any(
        tag in c for tag in ('_lag', '_roll_', '_dev_', '_diff_')
    )]

    print(f"Shape BEFORE dropna: {df.shape}")
    n_before = len(df)
    df = df.dropna(subset=window_feature_cols).reset_index(drop=True)
    n_after = len(df)
    print(f"Shape AFTER dropna (window features only): {df.shape}")
    print(f"Rows dropped: {n_before - n_after} ({(n_before-n_after)/n_before*100:.1f}%)")
    print("(root_cause / precursor_window are left as-is - they're metadata, not features)")
    
    if 'label_fail_6h' in df.columns:
        print(f"label_fail_6h positive rate AFTER dropna: {df['label_fail_6h'].mean():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved engineered features to {output_csv}")

if __name__ == "__main__":
    main()