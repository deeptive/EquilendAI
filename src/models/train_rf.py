import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_rf_tuned(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Search space to find the best settings for 0.88 accuracy
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [300, 500],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Search for the best parameters
    search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42)
    search.fit(X_resampled, y_resampled)
    
    print(f"Best RF Params: {search.best_params_}")
    return search.best_estimator_

if __name__ == "__main__":
    df = pd.read_csv(r"D:\EquilendAI\scripts\data\equilend_mock_data.csv")
    
    # Standardized Feature Engineering
    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]
    df["risk_proxy"] = (df["utility_bill_average"] / df["monthly_income"]) - (df["repayment_history_pct"] / 100)
    df["income_to_bill_ratio"] = df["monthly_income"] / df["utility_bill_average"]
    df["repayment_bill_interaction"] = df["repayment_history_pct"] * df["utility_bill_average"]
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("default_status", axis=1)
    y = df["default_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Correct Imputation
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = train_rf_tuned(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nRF Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"RF AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")