"""
TeleXAI Streamlit Dashboard
===========================
Interactive UI for network engineers to monitor 5G tower health and 
inspect explainable AI (SHAP/LIME) outputs for at-risk nodes.
"""

import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explain import TeleXAIExplainer

st.set_page_config(page_title="TeleXAI | NOC Dashboard", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# THEME
# Slate-navy NOC palette with a single cyan "signal" accent reused across the
# sidebar status dot, KPI card top-bars, and the telemetry chart's threshold
# line, standard red/amber/green are kept for risk severity since clarity
# matters more than novelty for that specific signal.
# =============================================================================
COLORS = {
    "bg": "#0A0E1A",
    "surface": "#111827",
    "surface_alt": "#1A2333",
    "border": "#26324A",
    "text": "#E8ECF4",
    "text_dim": "#8A96AD",
    "signal": "#22D3EE",
    "critical": "#EF4444",
    "elevated": "#F59E0B",
    "nominal": "#22C55E",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stToolbar"] {{visibility: hidden;}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}

.stApp {{
    background: {COLORS['bg']};
}}

[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}

[data-testid="stSidebar"] * {{
    color: {COLORS['text']};
}}

h1, h2, h3, h4 {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {COLORS['text']} !important;
    letter-spacing: -0.01em;
}}

p, span, div, label {{
    color: {COLORS['text']};
}}

hr {{
    border-color: {COLORS['border']};
}}

/* App header bar */
.telexai-header {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}}
.telexai-header .brand {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.6rem;
    letter-spacing: 0.04em;
    color: {COLORS['text']};
}}
.telexai-header .brand span {{
    color: {COLORS['signal']};
}}
.telexai-header .tagline {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {COLORS['text_dim']};
    font-size: 0.95rem;
}}

/* KPI cards */
.kpi-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    height: 100%;
}}
.kpi-card .kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLORS['text_dim']};
    margin-bottom: 0.4rem;
}}
.kpi-card .kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1.15;
}}
.kpi-card .kpi-sub {{
    font-size: 0.8rem;
    color: {COLORS['text_dim']};
    margin-top: 0.3rem;
}}
.kpi-card.accent-critical {{ border-top: 3px solid {COLORS['critical']}; box-shadow: 0 4px 20px rgba(239,68,68,0.15); }}
.kpi-card.accent-elevated {{ border-top: 3px solid {COLORS['elevated']}; box-shadow: 0 4px 20px rgba(245,158,11,0.15); }}
.kpi-card.accent-nominal  {{ border-top: 3px solid {COLORS['nominal']}; box-shadow: 0 4px 20px rgba(34,197,94,0.12); }}
.kpi-card.accent-signal   {{ border-top: 3px solid {COLORS['signal']}; }}

/* Sidebar status pill */
.status-pill {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: {COLORS['nominal']};
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 1rem;
}}
.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {COLORS['nominal']};
    box-shadow: 0 0 8px {COLORS['nominal']};
    animation: pulse 1.6s ease-in-out infinite;
}}
@keyframes pulse {{
    0%   {{ opacity: 1; }}
    50%  {{ opacity: 0.35; }}
    100% {{ opacity: 1; }}
}}

/* Tabs */
[data-testid="stTabs"] button {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}}

/* Bordered containers (SHAP/LIME panels) */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {COLORS['surface']};
    border-color: {COLORS['border']} !important;
}}
</style>
""", unsafe_allow_html=True)


def risk_tier(risk_prob):
    """Shared thresholds so every visual element agrees on severity."""
    if risk_prob > 0.5:
        return "critical", "CRITICAL"
    elif risk_prob > 0.2:
        return "elevated", "ELEVATED"
    return "nominal", "NOMINAL"


def kpi_card(label, value, sub="", accent="signal"):
    st.markdown(f"""
    <div class="kpi-card accent-{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{COLORS.get(accent, COLORS['text'])};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return TeleXAIExplainer(model_name="lightgbm")

@st.cache_data
def load_data(_engine):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(
        os.path.join(base_dir, 'data', 'processed', 'test_data.csv'),
        parse_dates=['timestamp']
    )

    # operational_label isn't saved in test_data.csv (it only ever lived in
    # a separate local script), so rebuild it here: any hour within 6h of
    # a real failure_event, matching generate_dataset.py's original
    # (pre-label-fix) definition. This is what makes the "within window
    # but signal hasn't started yet" message in the UI actually reachable.
    df = df.sort_values(['tower_id', 'timestamp']).reset_index(drop=True)
    df['operational_label'] = 0
    for tid, g in df.groupby('tower_id'):
        fail_times = g.loc[g['failure_event'] == 1, 'timestamp']
        for fail_time in fail_times:
            mask = (
                (df['tower_id'] == tid)
                & (df['timestamp'] <= fail_time)
                & (df['timestamp'] >= fail_time - pd.Timedelta(hours=6))
            )
            df.loc[mask, 'operational_label'] = 1

    # Score every row once so the UI can default to something worth
    # looking at, instead of whatever timestamp happens to sort first.
    risk_scores = _engine.model.predict_proba(df[_engine.feature_cols])[:, 1]
    df = pd.concat([df, pd.Series(risk_scores, name='risk_score', index=df.index)], axis=1)

    return df

engine = load_engine()
df = load_data(engine)

with st.sidebar:
    st.markdown("""
    <div class="status-pill">
        <div class="status-dot"></div>
        SYSTEM STATUS: ONLINE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📡 NOC Controls")
    st.caption("Select a cell tower telemetry log to inspect.")
    st.divider()

    failing_towers = df[df['label_fail_6h'] == 1]['tower_id'].unique()

    # Default to the tower with the single highest predicted risk, so the
    # dashboard opens on something worth explaining instead of a quiet hour.
    best_tower = df[df['tower_id'].isin(failing_towers)].sort_values('risk_score', ascending=False)['tower_id'].iloc[0]
    default_tower_idx = list(failing_towers).index(best_tower)

    st.markdown("**Tower**")
    selected_tower = st.selectbox("Select Tower ID", failing_towers, index=default_tower_idx, label_visibility="collapsed")

    tower_data = df[df['tower_id'] == selected_tower].sort_values('timestamp').reset_index(drop=True)
    default_ts_idx = int(tower_data['risk_score'].idxmax())

    st.markdown("**Timestamp**")
    selected_timestamp = st.selectbox("Select Timestamp", tower_data['timestamp'], index=default_ts_idx, label_visibility="collapsed")

    st.divider()
    st.caption("TeleXAI · Explainable Predictive Maintenance")
    st.caption("Model: LightGBM + TreeSHAP + LIME")

current_row = tower_data[tower_data['timestamp'] == selected_timestamp]
ground_truth = current_row['root_cause'].values[0]
is_precursor = current_row['precursor_window'].values[0]
is_operational_window = current_row['operational_label'].values[0]
risk_prob = engine.predict(current_row)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="telexai-header">
    <div class="brand">TELE<span>XAI</span></div>
    <div class="tagline">5G Network Health &amp; Explainability</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"<p style='color:{COLORS['text_dim']}; margin-top:-0.5rem;'>Explainable predictive maintenance for European telecom operations</p>", unsafe_allow_html=True)

# =============================================================================
# TOP KPI BANNER
# =============================================================================
tier, tier_label = risk_tier(risk_prob)

if is_operational_window and not is_precursor:
    context_value, context_sub, context_accent = "WITHIN WINDOW", "Pre-signal, telemetry still nominal", "elevated"
elif is_precursor:
    context_value, context_sub, context_accent = "PRECURSOR ACTIVE", f"Cause: {ground_truth}", "critical"
else:
    context_value, context_sub, context_accent = "NOMINAL", "No active failure signature", "nominal"

k1, k2, k3 = st.columns(3)
with k1:
    kpi_card("Current Risk Score", f"{risk_prob*100:.1f}%", tier_label, accent=tier)
with k2:
    kpi_card("Tower ID", selected_tower, f"as of {selected_timestamp}", accent="signal")
with k3:
    kpi_card("Timeline Context", context_value, context_sub, accent=context_accent)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# HISTORICAL TELEMETRY (last 24h leading up to the selected hour)
# =============================================================================
st.markdown("#### Historical Telemetry")
st.caption("Hardware temperature and packet loss over the 24 hours leading up to the selected timestamp.")

window_start = selected_timestamp - pd.Timedelta(hours=24)
history = tower_data[(tower_data['timestamp'] >= window_start) & (tower_data['timestamp'] <= selected_timestamp)]

# plotly.express doesn't support a true dual y-axis on a single line() call
# (each series would share one scale), so the figure is built with
# graph_objects + make_subplots instead, which is the correct tool for two
# differently-scaled metrics on one timeline. px is still used above for
# the qualitative color sequence to keep a consistent palette.
palette = px.colors.qualitative.Set2
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(
    x=history['timestamp'], y=history['hardware_temp_c'],
    name="Hardware Temp (°C)", mode="lines+markers",
    line=dict(color=COLORS['critical'], width=2.5), marker=dict(size=5),
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=history['timestamp'], y=history['packet_loss_pct'],
    name="Packet Loss (%)", mode="lines+markers",
    line=dict(color=COLORS['signal'], width=2.5, dash="dot"), marker=dict(size=5),
), secondary_y=True)

fig.add_vline(x=selected_timestamp, line_width=1.5, line_dash="dash", line_color=COLORS['text_dim'])

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=COLORS['surface'],
    plot_bgcolor=COLORS['surface'],
    font=dict(family="IBM Plex Sans", color=COLORS['text']),
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
fig.update_yaxes(title_text="Hardware Temp (°C)", secondary_y=False, showgrid=True, gridcolor=COLORS['border'])
fig.update_yaxes(title_text="Packet Loss (%)", secondary_y=True, showgrid=False)

st.plotly_chart(fig, width='stretch')

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# XAI LAYOUT
# =============================================================================
st.markdown("#### Why is the model making this prediction?")

tab_shap, tab_lime = st.tabs(["🔬  SHAP — Global/Local Attribution", "📊  LIME — Local Boundaries"])

with tab_shap:
    with st.container(border=True):
        st.caption("Shows how each feature pushes the risk probability up (red) or down (blue).")
        shap_obj = engine.get_shap_explanation(current_row)
        fig_shap, ax = plt.subplots(figsize=(9, 5))
        fig_shap.patch.set_facecolor(COLORS['surface'])
        ax.set_facecolor(COLORS['surface'])
        shap.plots.waterfall(shap_obj, show=False)
        st.pyplot(fig_shap)
        plt.clf()

with tab_lime:
    with st.container(border=True):
        st.caption("Shows strict local feature boundaries that contributed to this specific prediction.")
        lime_html = engine.get_lime_explanation(current_row)
        components.html(lime_html, height=500, scrolling=True)

st.markdown("---")
st.caption("Built for EU Telecom Predictive Maintenance Research | Architecture: LightGBM + TreeSHAP + LIME")