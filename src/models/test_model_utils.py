# src/models/test_model_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ✅ Correct import (project-root based)
from src.models.model_utils import (
    build_pipeline,
    train_pipeline,
    save_pipeline,
    load_pipeline
)


if __name__ == "__main__":
    print("🚀 Testing model_utils pipeline...\n")

    # Load data
    df = pd.read_csv(r"D:\EquilendAI\scripts\data\equilend_mock_data.csv")

    # 🔥 Feature engineering
    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)

    # Impute
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Define features
    numeric_features = [
        "monthly_income",
        "utility_bill_average",
        "repayment_history_pct",
        "bill_income_ratio"
    ]

    categorical_features = [
        "gender",
        "employment_length"
    ]

    X = df[numeric_features + categorical_features]
    y = df["default_status"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Build pipeline
    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        model_type="xgb",
        y_train=y_train
    )
    print("✔ Pipeline built")

    # ✅ Train
    pipeline = train_pipeline(pipeline, X_train, y_train)
    print("✔ Pipeline trained")

    # ✅ Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print("✔ Prediction works")

    print("\n📊 Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))

    # ✅ Save
    save_pipeline(pipeline)
    print("\n✔ Pipeline saved")

    # ✅ Load
    loaded_pipeline = load_pipeline()
    print("✔ Pipeline loaded")

    # ✅ Test loaded pipeline
    y_pred_loaded = loaded_pipeline.predict(X_test)
    print("✔ Loaded pipeline prediction works")

    print("\n🎉 ALL TESTS PASSED")