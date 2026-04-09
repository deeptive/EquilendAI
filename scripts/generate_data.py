import pandas as pd
import numpy as np
import random
import os

def generate_equilend_data(num_records=5000):
    np.random.seed(42)
    random.seed(42)

    data = []
    for _ in range(num_records):
        # 1. Protected Attribute (For Fairness/Bias Testing - Task 09)
        gender = random.choice(['Male', 'Female', 'Non-Binary'])
        
        # 2. Base Financials
        # Introduce a slight historical bias in income to give the Fairness report something to catch
        if gender == 'Female':
            income = int(np.random.normal(45000, 12000))
        else:
            income = int(np.random.normal(52000, 15000))
            
        income = max(15000, income) # Floor income
        
        # 3. Alternative Data (Utility Bills)
        utility_bill = int(np.random.normal(2500, 800))
        utility_bill = max(500, utility_bill)
        
        # 4. Repayment History (0 to 100%)
        repayment_history = min(100, max(0, int(np.random.normal(75, 20))))
        
        # 5. Employment Length
        employment_length = random.choice(['< 1 year', '1-3 years', '4-7 years', '8+ years'])
        
        # 6. Target Variable Logic (Default: 1, Paid: 0)
        # Create a logical pattern for XGBoost/Random Forest to learn
        risk_score = (utility_bill / income) * 100 - (repayment_history * 0.5)
        
        # Adjust threshold to create a Class Imbalance (~15-20% default rate) for SMOTE (Task 07)
        if risk_score > -25: 
            default_status = np.random.choice([1, 0], p=[0.7, 0.3]) # High risk
        else:
            default_status = np.random.choice([1, 0], p=[0.05, 0.95]) # Low risk
            
        data.append({
            'gender': gender,
            'monthly_income': income,
            'utility_bill_average': utility_bill,
            'repayment_history_pct': repayment_history,
            'employment_length': employment_length,
            'default_status': default_status
        })

    df = pd.DataFrame(data)

    # 7. Inject Missing Values for Task 03 (Iterative Imputation)
    # Randomly make 10% of utility bills and 5% of repayment histories NaN
    df.loc[df.sample(frac=0.10).index, 'utility_bill_average'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'repayment_history_pct'] = np.nan

    return df

if __name__ == "__main__":
    print("Generating EquiLend AI Synthetic Dataset...")
    df = generate_equilend_data(10000) # Generate 10k rows
    
    # Check default rate for SMOTE
    print("\nClass Imbalance Check (Target = default_status):")
    print(df['default_status'].value_counts(normalize=True) * 100)
    
    # Save to data folder
    os.makedirs('data', exist_ok=True)
    filepath = 'data/equilend_mock_data.csv'
    df.to_csv(filepath, index=False)
    print(f"\nDataset saved successfully to {filepath}")
