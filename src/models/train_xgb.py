import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def train_xgboost_88_push(X_train, y_train, X_val, y_val):
    # SMOTE is applied only to the training set
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Tuning parameters specifically for that last 0.2% boost
    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.01,       # Slowed down for higher precision
        max_depth=7,              # Slightly deeper to capture complex patterns
        min_child_weight=3,       # Higher value prevents overfitting to SMOTE noise
        gamma=0.3,                # Minimum loss reduction required to make a split
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,            # L1 regularization to prune features
        reg_lambda=2.0,           # L2 regularization to prevent weight explosion
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=100
    )

    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

if __name__ == "__main__":
    df = pd.read_csv(r"D:\EquilendAI\scripts\data\equilend_mock_data.csv")

    # Engineered features from your successful 0.878 run
    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]
    df["risk_proxy"] = (df["utility_bill_average"] / df["monthly_income"]) - (df["repayment_history_pct"] / 100)
    df["income_to_bill_ratio"] = df["monthly_income"] / df["utility_bill_average"]
    df["repayment_bill_interaction"] = df["repayment_history_pct"] * df["utility_bill_average"]
    
    # Clean up and Encode
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    df.columns = df.columns.str.replace(r"[<>\[\]]", "", regex=True)

    X = df.drop("default_status", axis=1)
    y = df["default_status"]

    # Split into Train, Validation, and Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    # Impute missing values based on Train only
    medians = X_train.median(numeric_only=True)
    X_train, X_val, X_test = X_train.fillna(medians), X_val.fillna(medians), X_test.fillna(medians)

    # Feature Scaling: Sometimes the final 0.2% is hidden in the feature scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = train_xgboost_88_push(X_train_scaled, y_train, X_val_scaled, y_val)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\nTarget Achieved? Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")