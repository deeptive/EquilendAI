import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.evaluation.fairness import check_model_fairness
from src.models.xgboost_model import load_artifact, predict_xgb


def run_bias_detection(
    model_path: str = "models/xgboost_model.joblib",
    data_path: str = "data/equilend_mock_data.csv",
    threshold: float = 0.10,
    output_path: str = "reports/fairness_report.json",
) -> Dict[str, Any]:
    """
    Load model + data, score predictions, and run fairness checks.

    Since the synthetic dataset currently lacks true `age` and `state` columns,
    this script creates deterministic placeholder groups so fairness plumbing
    can run end-to-end on dummy data.
    """
    artifact = load_artifact(model_path)
    df = pd.read_csv(data_path)

    # If the dataset does not contain these attributes yet, create placeholders
    # to keep the bias detection path executable for challenge workflows.
    if "age" not in df.columns:
        rng = np.random.default_rng(42)
        df["age"] = rng.integers(18, 70, size=len(df))
    if "state" not in df.columns:
        rng = np.random.default_rng(42)
        df["state"] = rng.choice(["CA", "TX", "NY", "FL", "WA"], size=len(df))

    # Model features are expected to be the raw columns used during training.
    feature_candidates = [
        "gender",
        "monthly_income",
        "utility_bill_average",
        "repayment_history_pct",
        "employment_length",
    ]
    X_new = df[feature_candidates].copy()
    y_true = df["default_status"].astype(int)

    y_proba = predict_xgb(artifact, X_new)
    y_pred = (y_proba >= 0.5).astype(int)

    fairness = check_model_fairness(
        df=df,
        y_true=y_true,
        y_pred=pd.Series(y_pred),
        age_col="age",
        state_col="state",
        max_diff=threshold,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fairness, f, indent=2)

    return fairness


if __name__ == "__main__":
    result = run_bias_detection()
    print("Bias detection complete.")
    print(json.dumps(result, indent=2))

