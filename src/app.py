import streamlit as st
import pandas as pd
import numpy as np
import time
import os

from models.train_rf import train_baseline_model, evaluate_model
from sklearn.model_selection import train_test_split

# Earthy / Professional Theme Colors
PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#5D4037"

# 🔹 Robust path (works everywhere)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "scripts", "data", "equilend_mock_data.csv")


def load_data():
    return pd.read_csv(DATA_PATH)


def preprocess(data):
    X = data.drop("default_status", axis=1)
    y = data["default_status"]

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())

    return X, y


def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")
    
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through alternative data.")

    # Sidebar
    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # 🔹 Load + preprocess once
    data = load_data()
    X, y = preprocess(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_baseline_model(X_train, y_train)

    if choice == "New Application":
        st.subheader("Manual Loan Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            income = st.number_input("Monthly Income (₹)", min_value=0)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)

        if st.button("Analyze Risk"):
            with st.spinner('AI Model Calculating...'):
                time.sleep(1)

                # 🔹 Input → dataframe
                input_data = pd.DataFrame({
                    "monthly_income": [income],
                    "utility_bill_average": [utility_bill],
                    "repayment_history_pct": [repayment_history],
                    "gender_Male": [1],
                    "gender_Non-Binary": [0],
                    "employment_length_1-3 years": [0],
                    "employment_length_4-7 years": [0],
                    "employment_length_8+ years": [0],
                })

                input_data = input_data.reindex(columns=X.columns, fill_value=0)

                prediction = model.predict(input_data)[0]
                risk_level = "High" if prediction == 1 else "Low"

                st.success(f"Analysis Complete for {name}")
                st.metric(label="Predicted Risk", value=risk_level)

                # 🔥 SHOW BASELINE METRICS HERE
                acc, prec, rec = evaluate_model(model, X_test, y_test)

                st.write("### 📊 Baseline Model Performance")

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.2f}")
                col2.metric("Precision", f"{prec:.2f}")
                col3.metric("Recall", f"{rec:.2f}")

    elif choice == "Dashboard":
        st.subheader("Model Performance Dashboard")

        acc, prec, rec = evaluate_model(model, X_test, y_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("Precision", f"{prec:.2f}")
        col3.metric("Recall", f"{rec:.2f}")

        chart_data = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall"],
            "Value": [acc, prec, rec]
        })
        st.bar_chart(chart_data.set_index("Metric"))

    elif choice == "Audit Logs":
        st.subheader("Audit Logs")
        st.info("No logs available yet.")


if __name__ == '__main__':
    main()