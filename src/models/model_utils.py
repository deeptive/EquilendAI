# src/models/model_utils.py

import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =========================
# PIPELINE BUILDER
# =========================

def build_pipeline(numeric_features, categorical_features, model_type="xgb", y_train=None):
    """
    Build full pipeline: preprocessing + SMOTE + model

    model_type: "xgb" or "rf"
    """

    # ✅ Preprocessing (NO scaling for tree models)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )

    # =========================
    #  MODEL SELECTION
    # =========================

    if model_type == "xgb":
        if y_train is None:
            raise ValueError("y_train is required for XGBoost to compute class imbalance")

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError("model_type must be 'xgb' or 'rf'")

    # =========================
    #  FULL PIPELINE
    # =========================

    pipeline = ImbPipeline(steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    return pipeline


# =========================
#  TRAIN
# =========================

def train_pipeline(pipeline, X_train, y_train):
    """
    Train pipeline
    """
    pipeline.fit(X_train, y_train)
    return pipeline


# =========================
# SAVE / LOAD
# =========================

def save_pipeline(pipeline, path="model/final_pipeline.pkl"):
    """
    Save trained pipeline
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path="model/final_pipeline.pkl"):
    """
    Load trained pipeline
    """
    return joblib.load(path)
