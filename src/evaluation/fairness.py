import pandas as pd
import numpy as np

def calculate_disparate_impact(df, group_col, privileged_val, unprivileged_val):
    # Approval rate = prediction is 0 (Low Risk)
    rates = df.groupby(group_col)['prediction'].apply(lambda x: (x == 0).mean())
    rate_priv = rates.get(privileged_val, 0)
    rate_unpriv = rates.get(unprivileged_val, 0)
    
    if rate_priv == 0: return 1.0
    return rate_unpriv / rate_priv

def run_bias_audit(model, X_test):
    df_audit = X_test.copy()
    df_audit['prediction'] = model.predict(X_test)
    metrics = {}

    # --- DEBUGGING: SEE ALL COLUMNS IN TERMINAL ---
    print("\n--- BIAS AUDIT DEBUG ---")
    print(f"Available Columns: {df_audit.columns.tolist()}")

    # 1. Gender Audit (Look for anything containing 'gender')
    gender_cols = [c for c in df_audit.columns if 'gender' in c.lower()]
    if gender_cols:
        target = gender_cols[0] # Usually 'gender_Male'
        metrics['Gender (Protected vs Privileged)'] = calculate_disparate_impact(df_audit, target, 1, 0)
        print(f"Audit matched Gender to: {target}")

    # 2. Age Audit (Look for anything containing 'age')
    age_cols = [c for c in df_audit.columns if 'age' in c.lower()]
    if age_cols:
        target_age = age_cols[0]
        df_audit['is_young'] = df_audit[target_age] < 25
        metrics['Age (Young vs Old)'] = calculate_disparate_impact(df_audit, 'is_young', False, True)
        print(f"Audit matched Age to: {target_age}")

    print("------------------------\n")
    return metrics