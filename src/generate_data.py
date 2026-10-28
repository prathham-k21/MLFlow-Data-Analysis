"""
FinTech Credit Default Risk Dataset Generator
Simulates realistic baseline (reference) and post-deployment production (current) datasets
exhibiting both Covariate Shift (Data Drift) and Concept Drift under macroeconomic stress.
"""

import os
import json
import numpy as np
import pandas as pd

def generate_credit_data(n_samples: int = 5000, is_drift: bool = False, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    if not is_drift:
        # Reference Data: Stable economic conditions
        annual_income = np.random.lognormal(mean=11.0, sigma=0.45, size=n_samples)  # Mean ~$65,000
        debt_to_income = np.random.normal(loc=18.5, scale=5.5, size=n_samples)      # Mean ~18.5%
        credit_score = np.random.normal(loc=710, scale=45, size=n_samples)          # Mean ~710
        loan_amount = np.random.normal(loc=15000, scale=6000, size=n_samples)       # Mean ~$15,000
        interest_rate = np.random.normal(loc=10.5, scale=2.5, size=n_samples)       # Mean ~10.5%
        employment_length = np.random.gamma(shape=3.0, scale=2.2, size=n_samples)   # Mean ~6.6 yrs
        revolving_balance = np.random.exponential(scale=8500, size=n_samples)       # Mean ~$8,500
        num_derogatory = np.random.poisson(lam=0.35, size=n_samples)
        
        home_ownership = np.random.choice(['MORTGAGE', 'RENT', 'OWN'], size=n_samples, p=[0.50, 0.38, 0.12])
        loan_purpose = np.random.choice(
            ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'education'],
            size=n_samples,
            p=[0.48, 0.25, 0.14, 0.08, 0.05]
        )
    else:
        # Current Data: Macroeconomic stress (Inflation, interest rate hikes, debt surge)
        # Covariate shift: Lower disposable income, higher DTI, lower credit score, higher rates
        annual_income = np.random.lognormal(mean=10.75, sigma=0.50, size=n_samples) # Mean ~$51,000 (Drifted)
        debt_to_income = np.random.normal(loc=26.8, scale=7.2, size=n_samples)      # Mean ~26.8% (Drifted)
        credit_score = np.random.normal(loc=658, scale=52, size=n_samples)          # Mean ~658 (Drifted)
        loan_amount = np.random.normal(loc=19800, scale=7200, size=n_samples)       # Mean ~$19,800 (Drifted)
        interest_rate = np.random.normal(loc=15.4, scale=3.2, size=n_samples)       # Mean ~15.4% (Drifted)
        employment_length = np.random.gamma(shape=2.4, scale=2.0, size=n_samples)   # Mean ~4.8 yrs (Drifted)
        revolving_balance = np.random.exponential(scale=14000, size=n_samples)      # Mean ~$14,000 (Drifted)
        num_derogatory = np.random.poisson(lam=0.72, size=n_samples)                # Higher delinquencies
        
        home_ownership = np.random.choice(['MORTGAGE', 'RENT', 'OWN'], size=n_samples, p=[0.40, 0.52, 0.08])
        loan_purpose = np.random.choice(
            ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'education'],
            size=n_samples,
            p=[0.56, 0.26, 0.08, 0.06, 0.04]
        )

    # Post-process ranges to realistic bounds
    annual_income = np.clip(annual_income, 15000, 250000)
    debt_to_income = np.clip(debt_to_income, 2.0, 55.0)
    credit_score = np.clip(credit_score, 300, 850)
    loan_amount = np.clip(loan_amount, 1000, 50000)
    interest_rate = np.clip(interest_rate, 4.0, 29.9)
    employment_length = np.clip(employment_length, 0, 30)
    revolving_balance = np.clip(revolving_balance, 0, 100000)
    num_derogatory = np.clip(num_derogatory, 0, 10)

    # Calculate default probability using logistic function
    # Concept drift is introduced when is_drift=True via increased sensitivity to DTI and interest rate
    income_norm = (annual_income - 60000) / 25000
    dti_norm = (debt_to_income - 20) / 6
    credit_norm = (credit_score - 700) / 50
    rate_norm = (interest_rate - 11) / 3
    derog_norm = num_derogatory * 0.5

    if not is_drift:
        logit = -2.1 + (0.55 * dti_norm) - (0.95 * credit_norm) - (0.40 * income_norm) + (0.45 * rate_norm) + (0.35 * derog_norm)
    else:
        # Concept Drift: Relationship shifts as economic squeeze elevates sensitivity to leverage
        logit = -1.6 + (0.85 * dti_norm) - (1.10 * credit_norm) - (0.50 * income_norm) + (0.65 * rate_norm) + (0.50 * derog_norm)

    prob_default = 1 / (1 + np.exp(-logit))
    # Clip probabilities
    prob_default = np.clip(prob_default, 0.01, 0.95)
    default_label = (np.random.binomial(n=1, p=prob_default)).astype(int)

    df = pd.DataFrame({
        'annual_income': np.round(annual_income, 2),
        'debt_to_income_ratio': np.round(debt_to_income, 2),
        'credit_score': np.round(credit_score, 1),
        'loan_amount': np.round(loan_amount, 2),
        'interest_rate': np.round(interest_rate, 2),
        'employment_length_years': np.round(employment_length, 1),
        'revolving_balance': np.round(revolving_balance, 2),
        'num_derogatory_records': num_derogatory.astype(int),
        'home_ownership': home_ownership,
        'loan_purpose': loan_purpose,
        'loan_default': default_label
    })
    
    return df

def save_datasets(output_dir: str = 'e:/MLDevOps/data'):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Reference (Baseline) Credit Dataset (5,000 records)...")
    ref_df = generate_credit_data(n_samples=5000, is_drift=False, random_seed=42)
    ref_path = os.path.join(output_dir, 'reference_data.csv')
    ref_df.to_csv(ref_path, index=False)
    print(f"Saved: {ref_path} | Default Rate: {ref_df['loan_default'].mean():.2%}")

    print("Generating Production (Current) Credit Dataset with Drift (5,000 records)...")
    cur_df = generate_credit_data(n_samples=5000, is_drift=True, random_seed=101)
    cur_path = os.path.join(output_dir, 'current_data.csv')
    cur_df.to_csv(cur_path, index=False)
    print(f"Saved: {cur_path} | Default Rate: {cur_df['loan_default'].mean():.2%}")

    metadata = {
        "dataset_name": "FinTech Lending & Credit Default Risk Dataset",
        "domain": "Financial Services / Credit Risk",
        "reference_records": len(ref_df),
        "current_records": len(cur_df),
        "features": list(ref_df.columns),
        "target": "loan_default",
        "reference_default_rate": float(ref_df['loan_default'].mean()),
        "current_default_rate": float(cur_df['loan_default'].mean())
    }
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Dataset generation and metadata saved successfully.")

if __name__ == '__main__':
    save_datasets()
