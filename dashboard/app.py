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

# Add the project root to the system path so we can import our inference engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explain import TeleXAIExplainer

# --- Configuration ---
st.set_page_config(page_title="TeleXAI NOC Dashboard", page_icon="📡", layout="wide")

# --- Caching Engine and Data ---
@st.cache_resource
def load_engine():
    return TeleXAIExplainer(model_name="lightgbm")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'test_data.csv'))

engine = load_engine()
df = load_data()

# --- UI Sidebar: NOC Controls ---
st.sidebar.title(" TeleXAI NOC Controls")
st.sidebar.markdown("Select a cell tower telemetry log to inspect.")

# Filter for towers that actually had a failure in the test set
failing_towers = df[df['label_fail_6h'] == 1]['tower_id'].unique()
selected_tower = st.sidebar.selectbox("Select Tower ID", failing_towers)

# Filter timestamps for the selected tower
tower_data = df[df['tower_id'] == selected_tower].sort_values('timestamp')
selected_timestamp = st.sidebar.selectbox("Select Timestamp", tower_data['timestamp'])

# Get the specific row
current_row = tower_data[tower_data['timestamp'] == selected_timestamp]
ground_truth = current_row['root_cause'].values[0]
is_precursor = current_row['precursor_window'].values[0]
is_operational_window = current_row['operational_label'].values[0] if 'operational_label' in current_row.columns else 0

# --- Main UI: Status and Alerts ---
st.title("5G Network Health & Explainability")
st.markdown("---")

# 1. Timeline Context (NEW)
if is_operational_window and not is_precursor:
    st.info("Timeline Context:** This tower is within 6 hours of a failure, but the physical degradation has not started yet. The model correctly sees healthy telemetry here.")
elif is_precursor:
    st.warning(f"Simulation Ground Truth:** This tower is actively experiencing a '{ground_truth}' precursor signal.")
else:
    st.success("Simulation Ground Truth:** This tower is physically healthy and operating normally.")

# 2. Prediction Inference
risk_prob = engine.predict(current_row)

col1, col2, col3 = st.columns(3)
with col1:
    if risk_prob > 0.5:
        st.error(f"CRITICAL RISK: {risk_prob*100:.1f}%")
    elif risk_prob > 0.2:
        st.warning(f"ELEVATED RISK: {risk_prob*100:.1f}%")
    else:
        st.success(f"MODEL PREDICTION: {risk_prob*100:.1f}% Risk")

with col2:
    st.metric("Tower ID", selected_tower)
    
with col3:
    st.metric("Timestamp", selected_timestamp)

st.markdown("---")

# --- UI: Explainability Layers ---
st.header("Why is the model making this prediction?")
col_shap, col_lime = st.columns(2)

# 3. SHAP Explanation
with col_shap:
    st.subheader("Global/Local: SHAP Waterfall")
    st.markdown("Shows how each feature pushes the risk probability up (red) or down (blue).")
    
    shap_obj = engine.get_shap_explanation(current_row)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_obj, show=False)
    st.pyplot(fig)
    plt.clf()

# 4. LIME Explanation
with col_lime:
    st.subheader("Local Bounds: LIME Bar Chart")
    st.markdown("Shows strict local feature boundaries that contributed to this specific prediction.")
    
    lime_html = engine.get_lime_explanation(current_row)
    components.html(lime_html, height=500, scrolling=True)

st.markdown("---")
st.caption("Built for EU Telecom Predictive Maintenance Research | Architecture: LightGBM + TreeSHAP + LIME")