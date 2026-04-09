"""
Task 13 — Threshold Optimizer
Sweep decision thresholds, compute per-threshold metrics, and surface
the optimal cut-point for several business objectives.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)


# ── Core sweep ────────────────────────────────────────────────────────────────

def sweep_thresholds(y_true, y_prob, step: float = 0.01) -> pd.DataFrame:
    """
    Evaluate the model at every threshold from 0.01 to 0.99.

    Args:
        y_true : array-like of true binary labels  (0 = paid, 1 = default)
        y_prob : array-like of predicted default probabilities
        step   : threshold increment (default 0.01)

    Returns:
        pd.DataFrame with one row per threshold and columns:
          threshold, precision, recall, f1, accuracy, specificity,
          approval_rate, tp, tn, fp, fn
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.round(np.arange(step, 1.0, step), 4)

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1           = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0)
        accuracy     = (tp + tn) / len(y_true)
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # "Approved" = predicted non-default (0)
        approval_rate = float((y_pred == 0).mean())

        records.append({
            "threshold":     round(float(t), 4),
            "precision":     round(precision,    4),
            "recall":        round(recall,       4),
            "f1":            round(f1,           4),
            "accuracy":      round(accuracy,     4),
            "specificity":   round(specificity,  4),
            "approval_rate": round(approval_rate,4),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        })

    return pd.DataFrame(records)


# ── Single-threshold metrics ──────────────────────────────────────────────────

def get_metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """
    Return a full metrics dict for one specific threshold value.

    Returns:
        dict with keys: threshold, precision, recall, f1, accuracy,
        specificity, approval_rate, tp, tn, fp, fn, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold":      threshold,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "accuracy":       float((tp + tn) / len(y_true)),
        "specificity":    specificity,
        "approval_rate":  float((y_pred == 0).mean()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm,
    }


# ── Optimal threshold search ──────────────────────────────────────────────────

OBJECTIVE_DESCRIPTIONS = {
    "f1":        "Maximize F1 Score — balanced precision/recall trade-off",
    "precision": "Maximize Precision — minimise false approvals (risk-averse lender)",
    "recall":    "Maximize Recall — catch as many defaulters as possible",
    "balanced":  "Balanced (G-Mean) — geometric mean of sensitivity & specificity",
    "profit":    "Maximize Profit — weighted business-value model",
}


def find_optimal_threshold(y_true, y_prob, objective: str = "f1") -> dict:
    """
    Find the threshold that maximises a given business objective.

    Args:
        y_true    : true binary labels
        y_prob    : predicted default probabilities
        objective : one of 'f1', 'precision', 'recall', 'balanced', 'profit'

    Returns:
        dict — same schema as get_metrics_at_threshold() plus the sweep df row
    """
    df = sweep_thresholds(y_true, y_prob).copy()

    if objective == "f1":
        best_idx = df["f1"].idxmax()

    elif objective == "precision":
        best_idx = df["precision"].idxmax()

    elif objective == "recall":
        best_idx = df["recall"].idxmax()

    elif objective == "balanced":
        # Geometric mean of sensitivity (recall) and specificity
        df["gmean"] = np.sqrt(df["recall"] * df["specificity"])
        best_idx = df["gmean"].idxmax()

    elif objective == "profit":
        # Business model (per applicant):
        #   TN  → correctly approved good borrower  → +1 revenue unit
        #   FP  → incorrectly approved defaulter    → -5 loss units
        #   TP  → correctly denied defaulter        →  0 (avoided loss)
        #   FN  → incorrectly denied good borrower  → -0.5 (missed revenue)
        df["profit_score"] = (
            df["tn"] * 1.0
            + df["fp"] * (-5.0)
            + df["tp"] * 0.0
            + df["fn"] * (-0.5)
        )
        best_idx = df["profit_score"].idxmax()

    else:
        raise ValueError(
            f"Unknown objective '{objective}'. "
            f"Choose from: {list(OBJECTIVE_DESCRIPTIONS)}"
        )

    row = df.loc[best_idx]
    return {
        "threshold":     float(row["threshold"]),
        "precision":     float(row["precision"]),
        "recall":        float(row["recall"]),
        "f1":            float(row["f1"]),
        "accuracy":      float(row["accuracy"]),
        "specificity":   float(row["specificity"]),
        "approval_rate": float(row["approval_rate"]),
        "tp": int(row["tp"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
    }
