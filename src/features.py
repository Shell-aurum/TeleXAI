"""
TeleXAI Feature Engineering
===========================

Transforms raw hourly telemetry into a rich feature set for predictive modeling.
Calculates rolling statistics and rate-of-change metrics while strictly 
grouping by tower_id to prevent data leakage between distinct cell sites.
"""

import pandas as pd
import numpy as np
import os

def create_time_features(df):
    """Encodes time as cyclical features (sin/cos) so the model understands 
    that hour 23 and hour 0 are next to each other."""
    
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    df['dow_sin'] = np.sin(df['dayofweek'] * (2. * np.pi / 7))
    df['dow_cos'] = np.cos(df['dayofweek'] * (2. * np.pi / 7))
    
    return df.drop(['hour', 'dayofweek'], axis=1)

def create_rolling_features(df, feature_cols):
    """Calculates trends (rolling mean), volatility (rolling std), and 
    spikes (rate of change) for each tower independently."""
    
    # Ensure data is strictly ordered by time within each tower
    df = df.sort_values(by=['tower_id', 'timestamp'])
    
    grouped = df.groupby('tower_id')[feature_cols]
    
    # 3-hour and 6-hour rolling windows
    for window in [3, 6]:
        # Rolling mean (Trend)
        rolling_mean = grouped.rolling(window).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f'{col}_roll_mean_{window}h' for col in feature_cols]
        
        # Rolling standard deviation (Volatility/Noise)
        rolling_std = grouped.rolling(window).std().reset_index(level=0, drop=True)
        rolling_std.columns = [f'{col}_roll_std_{window}h' for col in feature_cols]
        
        df = df.join(rolling_mean).join(rolling_std)
        
    # Rate of change (1-hour difference to catch sudden spikes)
    diff_1h = grouped.diff(1)
    diff_1h.columns = [f'{col}_diff_1h' for col in feature_cols]
    df = df.join(diff_1h)
    
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', 'raw', 'telemetry.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    output_csv = os.path.join(output_dir, 'features.csv')
    
    print(f"Loading raw telemetry from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    
    # Core telemetry metrics to engineer features from
    telemetry_cols = [
        'sinr_db', 'rsrp_dbm', 'packet_loss_pct', 'jitter_ms',
        'latency_ms', 'prb_utilization_pct', 'connected_users', 
        'hardware_temp_c', 'throughput_mbps'
    ]
    
    print("Engineering cyclical time features...")
    df = create_time_features(df)
    
    print("Engineering rolling and rate-of-change features by tower...")
    df = create_rolling_features(df, telemetry_cols)
    
    # The first few rows of every tower will have NaNs due to rolling calculations.
    # We must drop these so the ML model doesn't crash.
    df = df.dropna().reset_index(drop=True)
    
    print(f"Engineered dataset shape: {df.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved engineered features to {output_csv}")

if __name__ == "__main__":
    main()