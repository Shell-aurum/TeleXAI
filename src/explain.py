"""
TeleXAI Inference & Explanation Engine
======================================

Loads a specified trained model (LightGBM, XGBoost, or Random Forest) and 
initializes SHAP and LIME explainers into memory. Provides rapid inference 
and explanation generation for single telemetry rows to serve the frontend.
"""
import os
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular

class TeleXAIExplainer:
    def __init__(self, model_name = "lightgbm"):
        """
        Initializes the model and explainers for TeleXAI. Provides methods for inference and explanation generation.
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, 'data', 'processed')

        # Loading Model and Features
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")

        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(os.path.join(self.models_dir, 'feature_columns.joblib'))
        self.model_name = model_name

        # Loading Background data for the explainers
        # Lime and Shap need a background distribution to calculate feature importance
        # We sample 500 rows from the test set to keep memory light
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test_data.csv'))
        self.background_data = test_df[self.feature_cols].sample(n = min(500, len(test_df)), random_state = 42)

        # Initializing Explainers
        print(f"Initializing explainers for {model_name.upper()}....")
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = self.background_data.values,
            feature_names=self.feature_cols,
            class_names = ['Healthy', 'Failure Risk'],
            mode='classification',
            random_state = 42
        )
        print("Models and Explainers and Initialized...")

    def predict(self, input_data: pd.DataFrame):
        """Returns probability of failure (class 1)"""
        # Ensure correct column order
        input_data = input_data[self.feature_cols]
        proba = self.model.predict_proba(input_data)[0][1]
        return proba

    def get_shap_explanation(self, input_data: pd.DataFrame):
        """
        Returns a structured SHAP explanation object specifically formatted 
        for plotting in Streamlit.
        """

        input_data = input_data[self.feature_cols]
        shap_values = self.shap_explainer(input_data)

        # Handle shape differences across algorithms (RF vs LGBM/XGB)
        if len(shap_values.shape) == 3: 
            # Multi-class output (like Random Forest) -> index 1 for 'Failure'
            return shap_values[0, :, 1]
        else:
            # Binary output (LightGBM / XGBoost)
            return shap_values[0]

    def get_lime_explanation(self, input_data: pd.DataFrame):
        """
        Generates a LIME explanation and returns the raw HTML string 
        so Streamlit can render it instantly.
        """
        input_data = input_data[self.feature_cols]
        row_values = input_data.iloc[0].values
        
        exp = self.lime_explainer.explain_instance(
            data_row=row_values, 
            predict_fn=self.model.predict_proba, 
            num_features=5
        )
        return exp.as_html()

if __name__ == "__main__":
    # Test the engine initialization and inference
    engine = TeleXAIExplainer(model_name="lightgbm")
    
    # Grab a single failing row from the test set to simulate real-time inference
    test_df = pd.read_csv(os.path.join(engine.data_dir, 'test_data.csv'))
    sample_row = test_df[test_df['label_fail_6h'] == 1].iloc[[0]]
    
    risk_score = engine.predict(sample_row)
    print(f"\nPredicted Failure Risk: {risk_score * 100:.1f}%")
    print("\nSHAP and LIME objects generated successfully.")