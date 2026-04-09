import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r'D:\EquilendAI\scripts\data\equilend_mock_data.csv')

# Basic preprocessing
df.fillna(df.median(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("default_status", axis=1)
y = df["default_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def train_baseline_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Train model
model = train_baseline_model(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("Model trained successfully ✅")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))