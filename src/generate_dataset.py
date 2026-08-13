"""
TeleXAI Synthetic 5G Telemetry Generator
=========================================

Generates hourly telemetry for a fleet of simulated cell towers, with
realistic daily/weekly seasonality, tower-level heterogeneity, and
INJECTED failure events with known ground-truth root causes.

Why synthetic, and why this design:
- Real telecom telemetry is proprietary and unavailable to students.
- Because we control the injection, we know the TRUE cause of every
  failure. This lets us later check whether SHAP/LIME explanations
  actually point at the real cause, instead of just "looking plausible."

Failure modes modeled (each has a distinct precursor signature that
appears BEFORE the failure, so a predictive model has a real signal
to learn from):

1. thermal_overload   - hardware_temp_c climbs steadily over 6-12h,
                         usually on already-hot, high-load towers.
2. signal_interference- sinr_db drops sharply and jitter/packet_loss
                         rise within a short (1-4h) window.
3. congestion_collapse- prb_utilization_pct and connected_users spike,
                         throughput and latency degrade together.
4. hardware_degradation - slow multi-day drift in jitter/packet_loss,
                         no single sharp trigger (harder to catch early).

Output: telemetry.csv with one row per (tower, hour).
Ground-truth failure-cause columns are included so YOU can validate
explanations, but should be dropped from the model's training features
(see the "columns to exclude from features" note in the README).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_TOWERS = 15
N_DAYS = 60
FREQ_HOURS = 1
SEED = 42
PREDICTION_HORIZON_H = 6   # "will this tower fail in the next 6 hours?"

FAILURE_TYPES = [
    "thermal_overload",
    "signal_interference",
    "congestion_collapse",
    "hardware_degradation",
]

rng = np.random.default_rng(SEED)


@dataclass
class TowerProfile:
    tower_id: str
    base_sinr: float
    base_rsrp: float
    base_temp: float
    base_users: float
    noise_scale: float
    urban: bool  # urban towers see heavier daily load swings


def make_tower_profiles(n=N_TOWERS):
    profiles = []
    for i in range(n):
        urban = rng.random() < 0.5
        profiles.append(TowerProfile(
            tower_id=f"TWR-{i+1:03d}",
            base_sinr=rng.uniform(14, 22),
            base_rsrp=rng.uniform(-95, -80),
            base_temp=rng.uniform(28, 36),
            base_users=rng.uniform(150, 400) if urban else rng.uniform(40, 150),
            noise_scale=rng.uniform(0.8, 1.3),
            urban=urban,
        ))
    return profiles


# ---------------------------------------------------------------------------
# Baseline (healthy) telemetry
# ---------------------------------------------------------------------------

def seasonal_load_factor(timestamps):
    """Daily + weekly load pattern: busier on weekday evenings, quieter at night."""
    hour = timestamps.hour + timestamps.minute / 60
    dow = timestamps.dayofweek  # 0=Mon
    daily = 0.5 + 0.5 * np.sin((hour - 8) / 24 * 2 * np.pi - np.pi / 2)
    daily = np.clip(daily, 0.15, 1.0)
    weekend_boost = np.where(dow >= 5, 1.15, 1.0)
    return daily * weekend_boost


def generate_baseline(profile: TowerProfile, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(timestamps)
    load = seasonal_load_factor(timestamps)
    load_mult = 1.6 if profile.urban else 1.0

    connected_users = profile.base_users * load * load_mult
    connected_users *= rng.normal(1.0, 0.05 * profile.noise_scale, n)
    connected_users = np.clip(connected_users, 5, None)

    prb_utilization = 20 + 70 * (connected_users / connected_users.max())
    prb_utilization += rng.normal(0, 3 * profile.noise_scale, n)
    prb_utilization = np.clip(prb_utilization, 5, 99)

    sinr = profile.base_sinr - 0.04 * (prb_utilization - 40) \
        + rng.normal(0, 1.2 * profile.noise_scale, n)
    rsrp = profile.base_rsrp + rng.normal(0, 1.5 * profile.noise_scale, n)

    hardware_temp = profile.base_temp + 6 * (prb_utilization / 100) \
        + 3 * np.sin((timestamps.hour - 14) / 24 * 2 * np.pi) \
        + rng.normal(0, 0.8 * profile.noise_scale, n)

    packet_loss = 0.05 + 0.02 * (prb_utilization / 100) \
        + np.abs(rng.normal(0, 0.05 * profile.noise_scale, n))
    packet_loss = np.clip(packet_loss, 0, None)

    jitter = 2 + 0.05 * (prb_utilization) + np.abs(rng.normal(0, 0.6 * profile.noise_scale, n))
    latency = 8 + 0.15 * prb_utilization + np.abs(rng.normal(0, 1.0 * profile.noise_scale, n))

    throughput = np.clip(220 - 1.6 * prb_utilization, 15, None) \
        + rng.normal(0, 4 * profile.noise_scale, n)
    throughput = np.clip(throughput, 5, None)

    return pd.DataFrame({
        "timestamp": timestamps,
        "tower_id": profile.tower_id,
        "urban": profile.urban,
        "sinr_db": sinr,
        "rsrp_dbm": rsrp,
        "packet_loss_pct": packet_loss,
        "jitter_ms": jitter,
        "latency_ms": latency,
        "prb_utilization_pct": prb_utilization,
        "connected_users": connected_users.round().astype(int),
        "hardware_temp_c": hardware_temp,
        "throughput_mbps": throughput,
    })


# ---------------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------------

def inject_thermal_overload(df, idx, ramp_hours):
    start = max(0, idx - ramp_hours)
    ramp = np.linspace(0, 1, idx - start + 1) ** 1.5
    df.loc[start:idx, "hardware_temp_c"] += ramp * rng.uniform(18, 28)
    df.loc[start:idx, "prb_utilization_pct"] += ramp * rng.uniform(10, 20)
    df.loc[start:idx, "throughput_mbps"] -= ramp * rng.uniform(20, 40)
    return start


def inject_signal_interference(df, idx, ramp_hours):
    start = max(0, idx - ramp_hours)
    ramp = np.linspace(0, 1, idx - start + 1)
    df.loc[start:idx, "sinr_db"] -= ramp * rng.uniform(10, 18)
    df.loc[start:idx, "packet_loss_pct"] += ramp * rng.uniform(3, 7)
    df.loc[start:idx, "jitter_ms"] += ramp * rng.uniform(8, 15)
    return start


def inject_congestion_collapse(df, idx, ramp_hours):
    start = max(0, idx - ramp_hours)
    ramp = np.linspace(0, 1, idx - start + 1) ** 1.3
    df.loc[start:idx, "prb_utilization_pct"] = np.clip(
        df.loc[start:idx, "prb_utilization_pct"] + ramp * rng.uniform(15, 25), 0, 100)
    df.loc[start:idx, "connected_users"] = (
        df.loc[start:idx, "connected_users"] * (1 + ramp * rng.uniform(0.4, 0.9))
    ).round().astype(int)
    df.loc[start:idx, "latency_ms"] += ramp * rng.uniform(30, 60)
    df.loc[start:idx, "throughput_mbps"] -= ramp * rng.uniform(30, 60)
    return start


def inject_hardware_degradation(df, idx, ramp_hours):
    # slower, noisier drift over a longer window - the "hard to catch" case
    start = max(0, idx - ramp_hours)
    n = idx - start + 1
    drift = np.linspace(0, 1, n) + rng.normal(0, 0.08, n).cumsum() / n
    drift = np.clip(drift, 0, None)
    df.loc[start:idx, "jitter_ms"] += drift * rng.uniform(6, 12)
    df.loc[start:idx, "packet_loss_pct"] += drift * rng.uniform(1.5, 4)
    df.loc[start:idx, "hardware_temp_c"] += drift * rng.uniform(4, 8)
    return start


INJECTORS = {
    "thermal_overload": (inject_thermal_overload, (6, 14)),
    "signal_interference": (inject_signal_interference, (1, 4)),
    "congestion_collapse": (inject_congestion_collapse, (2, 8)),
    "hardware_degradation": (inject_hardware_degradation, (24, 72)),
}


def inject_failures_for_tower(df: pd.DataFrame, n_events: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    n = len(df)
    df["failure_event"] = 0
    df["root_cause"] = None
    df["precursor_window"] = False
    df[f"label_fail_{PREDICTION_HORIZON_H}h"] = 0

    min_gap = 96  # hours between failure events, avoid overlap
    candidate_idxs = list(range(150, n - 5))
    rng.shuffle(candidate_idxs)

    placed = []
    for idx in candidate_idxs:
        if len(placed) >= n_events:
            break
        if all(abs(idx - p) > min_gap for p in placed):
            placed.append(idx)

    for idx in placed:
        failure_type = rng.choice(FAILURE_TYPES)
        injector, (lo, hi) = INJECTORS[failure_type]
        ramp_hours = int(rng.integers(lo, hi + 1))
        start = injector(df, idx, ramp_hours)

        df.loc[idx, "failure_event"] = 1
        df.loc[start:idx, "root_cause"] = failure_type
        df.loc[start:idx, "precursor_window"] = True

        # label = 1 for any hour within PREDICTION_HORIZON_H of the failure
        label_start = max(0, idx - PREDICTION_HORIZON_H)
        df.loc[label_start:idx, f"label_fail_{PREDICTION_HORIZON_H}h"] = 1

    # keep physical values sane after injection
    df["sinr_db"] = df["sinr_db"].clip(-15, 30)
    df["prb_utilization_pct"] = df["prb_utilization_pct"].clip(0, 100)
    df["packet_loss_pct"] = df["packet_loss_pct"].clip(0, 100)
    df["throughput_mbps"] = df["throughput_mbps"].clip(1, None)
    df["hardware_temp_c"] = df["hardware_temp_c"].clip(15, 95)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamps = pd.date_range("2026-01-01", periods=N_DAYS * 24, freq=f"{FREQ_HOURS}h")
    profiles = make_tower_profiles()

    all_towers = []
    for profile in profiles:
        base = generate_baseline(profile, timestamps)
        n_events = int(rng.integers(1, 4))  # 1-3 failures per tower over 60 days
        with_failures = inject_failures_for_tower(base, n_events)
        all_towers.append(with_failures)

    full = pd.concat(all_towers, ignore_index=True)
    full = full.sort_values(["tower_id", "timestamp"]).reset_index(drop=True)

    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "data", "raw", "telemetry.csv")
    full.to_csv(out_path, index=False)

    n_failures = full["failure_event"].sum()
    n_precursor_rows = full["precursor_window"].sum()
    print(f"Rows: {len(full):,} | Towers: {N_TOWERS} | Days: {N_DAYS}")
    print(f"Failure events: {n_failures} | Precursor-labeled rows: {n_precursor_rows:,}")
    print(full["root_cause"].value_counts(dropna=True))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()