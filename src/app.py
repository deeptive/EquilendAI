import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# Model and Evaluation imports
from models.train_rf import train_baseline_model, evaluate_model
from sklearn.model_selection import train_test_split
from evaluation.explainer import generate_shap_explanation 
from evaluation.fairness import run_bias_audit  

# --- THE WORKING PATH ---
DATA_PATH = r"C:\Users\Purva\OneDrive\Desktop\innovation_test\EquilendAI\EquilendAI\scripts\data\equilend_mock_data.csv"

def load_data():
    """Tries the working path to load the dataset."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"🚨 **Data file not found!** App is looking here: {DATA_PATH}")
        return pd.DataFrame()

def preprocess(data):
    """Cleans data and handles feature engineering for the AI Engine."""
    if data.empty:
        return data, None
    
    # Preprocessing Guardrails (Handling Task 03/05 requirements)
    data['monthly_income'] = data['monthly_income'].clip(lower=0)
    data = data.fillna(data.mean(numeric_only=True))
    
    # Feature Engineering
    data['bill_to_income_ratio'] = data['utility_bill_average'] / (data['monthly_income'] + 1)
    
    X = data.drop("default_status", axis=1)
    y = data["default_status"]

    # Categorical Encoding (This creates 'gender_Male', 'gender_Female', etc.)
    X = pd.get_dummies(X, drop_first=True)
    
    # Sanitize column names for model compatibility
    X.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X.columns]

    return X, y

def main():
    st.set_page_config(page_title="EquiLend AI - Decision Explainer", layout="wide")
    
    st.title("⚖️ EquiLend AI: Credit Scoring")
    st.markdown("### Transparent & Fair AI for Alternative Data Lending")

    st.sidebar.title("Control Panel")
    menu = ["New Application", "Dashboard"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # 1. Load and Process Data
    data = load_data()
    if data.empty:
        return

    X, y = preprocess(data)

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Training (Cached for performance)
    @st.cache_resource
    def get_trained_assets(_X_train, _y_train):
        rf = train_baseline_model(_X_train, _y_train)
        return rf

    rf_model = get_trained_assets(X_train, y_train)

    # --- UI LOGIC ---
    if choice == "New Application":
        st.subheader("Manual Loan Application")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value="Jane Doe")
            age = st.number_input("Age", min_value=18, max_value=120, value=30)
            income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0, value=2000)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 90)

        if st.button("Analyze Risk"):
            with st.spinner('AI Decision Engine Processing...'):
                time.sleep(0.3)

                # Prepare input matching training columns
                input_data = pd.DataFrame({
                    "monthly_income": [income],
                    "utility_bill_average": [utility_bill],
                    "repayment_history_pct": [repayment_history],
                    "bill_to_income_ratio": [utility_bill / (income + 1)]
                }).reindex(columns=X.columns, fill_value=0)

                # Prediction
                prediction = rf_model.predict(input_data)[0]
                prob = rf_model.predict_proba(input_data)[0][1]
                risk_level = "High" if prediction == 1 else "Low"

                st.success(f"Analysis Complete for {name}")
                m1, m2 = st.columns(2)
                m1.metric("Predicted Risk", risk_level)
                m2.metric("Default Probability", f"{prob*100:.1f}%")

                # SHAP Explanation (Task 08)
                with st.expander("🔍 Why this decision? (XAI Breakdown)"):
                    st.write("Generating factor analysis...")
                    fig = generate_shap_explanation(rf_model, input_data)
                    st.pyplot(fig) # Now receives Figure object
                    st.info("**Red** bars increase risk, **Blue** bars decrease risk.")

                if risk_level == "High":
                    st.error("🚨 Flagged for review based on risk profile.")
                else:
                    st.balloons()
                    st.success("✅ Application approved.")

    elif choice == "Dashboard":
        st.subheader("📊 Performance & Ethics Dashboard")
        
        # 1. Performance Metrics
        rf_res = evaluate_model(rf_model, X_test, y_test)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{rf_res[0]:.2f}")
        c2.metric("Precision", f"{rf_res[1]:.2f}")
        c3.metric("Recall", f"{rf_res[2]:.2f}")
        
        st.markdown("---")
        
        # 2. Bias Detection Audit (Task 09)
        st.subheader("🛡️ Bias Detection Audit (80% Rule)")
        st.info("The audit compares approval rates between groups to detect systemic discrimination.")
        
        bias_results = run_bias_audit(rf_model, X_test)
        
        if not bias_results:
            st.error("❌ **No Audit Data Generated!**")
            st.warning("Audit engine could not find protected columns (Gender/Age) in X_test.")
            with st.expander("Debug: View Columns in Dataset"):
                st.write("The audit is looking for words like 'gender' or 'age' in these columns:")
                st.write(X_test.columns.tolist())
        else:
            for group, ratio in bias_results.items():
                b_col1, b_col2 = st.columns([1, 4])
                b_col1.metric(label=group, value=f"{ratio:.2f}")
                
                if ratio < 0.8:
                    b_col2.error(f"⚠️ **Bias Warning:** {group} fails the 80% rule (Ratio < 0.80).")
                else:
                    b_col2.success(f"✅ **Fairness Passed:** {group} is within acceptable limits.")

if __name__ == '__main__':
    main()