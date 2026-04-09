# Task 05: Baseline Random Forest for Risk Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train_baseline_model(X_train, y_train):
    """
    Train a baseline Random Forest model.
    Note: Uses class_weight to handle imbalance while staying simple.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, precision, and recall.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    return accuracy, precision, recall


# 🔹 Optional standalone test (run this file directly)
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # load dataset
    data = pd.read_csv("../data/equilend_mock_data.csv")

    X = data.drop("default_status", axis=1)
    y = data["default_status"]

    # preprocessing
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train + evaluate
    model = train_baseline_model(X_train, y_train)
    acc, prec, rec = evaluate_model(model, X_test, y_test)

    print("=== BASELINE MODEL PERFORMANCE ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")