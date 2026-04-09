import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib

from src.preprocessing.data_cleaning import load_and_clean
from src.preprocessing.feature_encoding import encode_categorical_features
from src.evaluation.shap_analysis import (
    compute_shap_values,
    shap_feature_importance_bar_streamlit,
    shap_single_prediction_force_plot_streamlit,
)
from src.models.xgboost_model import load_artifact, predict_default_probability

# Earthy / Professional Theme Colors
PRIMARY_COLOR = "#2E7D32"  # Forest Green
ACCENT_COLOR = "#5D4037"  # Soil Brown

MODEL_PATH = "models/xgboost_model.joblib"
DATA_PATH = "data/equilend_mock_data.csv"


@st.cache_resource
def load_model_and_preprocessor():
    """
    Load the trained XGBoost artifact (model + scaler + feature columns).

    The artifact is produced by `src/models/xgboost_model.py` and ensures
    inference uses the exact same preprocessing metadata as training.
    """
    try:
        artifact = load_artifact(MODEL_PATH)
    except Exception as e:
        # If anything fails (e.g., model file missing), surface the error in the UI.
        st.error(f"Failed to load model artifact: {e}")
        return None

    return artifact

def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")
    
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through alternative data.")

    # Sidebar for Navigation
    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "New Application":
        st.subheader("Manual Loan Application")

        artifact = load_model_and_preprocessor()
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            income = st.number_input("Monthly Income (₹)", min_value=0)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
            employment_length = st.selectbox(
                "Employment Length",
                ["< 1 year", "1-3 years", "4-7 years", "8+ years"],
            )

        if artifact is None:
            st.warning("Model not available. Please train the model before running predictions.")
            return

        if st.button("Analyze Risk"):
            # LOGICAL ERRORS 1, 2, and 3 are hidden in this block
            with st.spinner('AI Model Calculating...'):
                time.sleep(1) # Simulate processing
                
                # Build raw features from user input using the same schema as training.
                # Note: "age" is currently collected for UX, but the synthetic dataset
                # used in training does not contain age/state columns, so it is not
                # included in the model feature set.
                input_data = {
                    "gender": gender,
                    "monthly_income": income,
                    "utility_bill_average": utility_bill,
                    "repayment_history_pct": repayment_history,
                    "employment_length": employment_length,
                }
                default_proba = predict_default_probability(artifact, input_data)

                # Higher probability of default implies higher risk.
                risk_level = "High" if default_proba >= 0.5 else "Low"

                st.success(f"Analysis Complete for {name}")
                st.metric(
                    label="Predicted Default Probability",
                    value=f"{default_proba:.2%}",
                )
                st.write(f"Recommended Decision: **{risk_level} Risk**")

                # SHAP-based explanation for this specific prediction.
                with st.expander("Explain this prediction (SHAP)"):
                    # Build a DataFrame matching the model's feature space for SHAP.
                    # We re-run encoding + scaling to obtain the exact feature vector
                    # that was fed into the model.
                    user_df = pd.DataFrame([input_data])
                    user_encoded = encode_categorical_features(user_df)
                    for col in artifact.feature_cols:
                        if col not in user_encoded.columns:
                            user_encoded[col] = 0.0
                    user_encoded = user_encoded[artifact.feature_cols]
                    user_scaled = artifact.scaler.transform(user_encoded)
                    user_scaled_df = pd.DataFrame(user_scaled, columns=artifact.feature_cols)

                    explainer, shap_values = compute_shap_values(artifact.model, user_scaled_df)

                    # Global-style bar plot for this single prediction (mean |SHAP|).
                    shap_feature_importance_bar_streamlit(
                        shap_values,
                        user_scaled_df,
                        title="Feature Importance (SHAP values)",
                    )

                    # Local force plot showing how each feature influenced this decision.
                    shap_single_prediction_force_plot_streamlit(
                        explainer,
                        shap_values,
                        user_scaled_df,
                        instance_index=0,
                        title="Contribution of Each Feature",
                    )

                # LOGICAL ERROR 4: Data is not saved anywhere yet

    elif choice == "Dashboard":
        st.subheader("Lender Rules Engine Overview")
        # Placeholder for visual charts
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Approved', 'Rejected', 'Pending'])
        st.line_chart(chart_data)

        st.markdown("### Dummy Predictions (XGBoost)")
        artifact = load_model_and_preprocessor()
        if artifact is None:
            st.warning("Model not available. Train the XGBoost model to see predictions.")
        else:
            try:
                df = load_and_clean(DATA_PATH)
                # Show a few sample predictions to prove end-to-end works.
                sample = df.head(5).copy()
                X_raw = sample.drop(columns=["default_status"])

                # Encode and align columns to training.
                X_enc = encode_categorical_features(X_raw)
                for col in artifact.feature_cols:
                    if col not in X_enc.columns:
                        X_enc[col] = 0.0
                X_enc = X_enc[artifact.feature_cols]
                X_scaled = artifact.scaler.transform(X_enc)

                sample["pred_default_proba"] = artifact.model.predict_proba(X_scaled)[:, 1]
                st.dataframe(
                    sample[
                        [
                            "monthly_income",
                            "utility_bill_average",
                            "repayment_history_pct",
                            "gender",
                            "employment_length",
                            "default_status",
                            "pred_default_proba",
                        ]
                    ]
                )
                st.caption(
                    f"Model test AUC (from training artifact): {artifact.test_auc:.4f}"
                )
            except Exception as e:
                st.error(f"Failed to generate dummy predictions: {e}")

if __name__ == '__main__':
    main()
