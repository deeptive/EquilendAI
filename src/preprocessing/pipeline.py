
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline():
    numeric_features = ['monthly_income', 'utility_bill_average', 'repayment_history_pct']
    categorical_features = ['gender', 'employment_length']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return Pipeline(steps=[('preprocessor', preprocessor)])