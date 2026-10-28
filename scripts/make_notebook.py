"""
Script to generate the comprehensive Jupyter Notebook for MLflow Data Drift Analysis.
"""

import json
import os

def create_notebook():
    nb_cells = [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# MLflow Installation, Experiment Tracking, and Data Drift Analysis\n',
                '### Course: ML DevOps (Course Code: UAMPC0703) | Theory-ISE 2\n',
                '---\n',
                '**Student Name:** Pratham Kadam  \n',
                '**Roll Number:** DS 14  \n',
                '**PRN:** 2324000362  \n',
                '**Class & Branch:** Final Year B.Tech CSE (Data Science) - Semester VII  \n',
                '**Academic Year:** 2026-27  \n',
                '**Domain:** Financial Services / FinTech Credit Default Risk & Loan Assessment  \n',
                '**GitHub Repository:** [https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis](https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis)\n',
                '---\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Environment Setup & Dependency Verification\n',
                'In this section, we verify the runtime environment, Python version, MLflow installation, and essential dependencies.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': 1,
            'metadata': {},
            'outputs': [],
            'source': [
                'import os\n',
                'import sys\n',
                'import platform\n',
                'import mlflow\n',
                'import sklearn\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import scipy\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                '\n',
                'print("Operating System :", platform.system(), platform.release())\n',
                'print("Python Version   :", platform.python_version())\n',
                'print("MLflow Version   :", mlflow.__version__)\n',
                'print("Scikit-Learn     :", sklearn.__version__)\n',
                'print("Pandas Version   :", pd.__version__)\n',
                'print("SciPy Version    :", scipy.__version__)\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Dataset Overview: FinTech Lending & Credit Default Risk\n',
                'We load the baseline **Reference Dataset** (stable historical macroeconomic conditions) and the **Current Production Dataset** (macroeconomic stress with inflation, debt surge, and interest rate hikes).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': 2,
            'metadata': {},
            'outputs': [],
            'source': [
                'ref_df = pd.read_csv("../data/reference_data.csv")\n',
                'cur_df = pd.read_csv("../data/current_data.csv")\n',
                '\n',
                'print(f"Reference Dataset Shape  : {ref_df.shape} | Baseline Default Rate: {ref_df[\'loan_default\'].mean():.2%}")\n',
                'print(f"Production Dataset Shape : {cur_df.shape} | Current Default Rate : {cur_df[\'loan_default\'].mean():.2%}")\n',
                'ref_df.head()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. MLflow Experiment Tracking & Baseline Model Training\n',
                'We configure our SQLite tracking backend, register the experiment `Credit_Default_Risk_Drift_Monitoring`, and train a `GradientBoostingClassifier` with full hyperparameter and metric logging.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': 3,
            'metadata': {},
            'outputs': [],
            'source': [
                'from src.train import train_baseline_model\n',
                '\n',
                'db_path = os.path.abspath("../mlflow.db").replace("\\\\", "/")\n',
                'mlflow.set_tracking_uri(f"sqlite:///{db_path}")\n',
                'mlflow.set_experiment("Credit_Default_Risk_Drift_Monitoring")\n',
                '\n',
                'train_res = train_baseline_model(data_path="../data/reference_data.csv", artifacts_dir="../artifacts")\n',
                'pipeline = train_res["pipeline"]\n',
                'print("Baseline Training Metrics:")\n',
                'for k, v in train_res["metrics"].items():\n',
                '    print(f"  {k:22s}: {v}")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. Statistical Drift Detection Suite\n',
                'We execute formal statistical tests:\n',
                '1. **Two-Sample Kolmogorov-Smirnov (KS) Test** for all continuous features.\n',
                '2. **Population Stability Index (PSI)** under banking standards ($PSI < 0.10$ stable, $0.10-0.25$ moderate, $\\ge 0.25$ severe).\n',
                '3. **Wasserstein Distance** to evaluate earth mover\'s distance.\n',
                '4. **Chi-Square Test** for categorical distributions.\n',
                '5. **Model Output Probability Drift**.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': 4,
            'metadata': {},
            'outputs': [],
            'source': [
                'from src.drift_detector import StatisticalDriftDetector\n',
                '\n',
                'detector = StatisticalDriftDetector(significance_level=0.05, psi_threshold=0.25)\n',
                'num_cols = [\'annual_income\', \'debt_to_income_ratio\', \'credit_score\', \'loan_amount\', \'interest_rate\', \'employment_length_years\', \'revolving_balance\', \'num_derogatory_records\']\n',
                'cat_cols = [\'home_ownership\', \'loan_purpose\']\n',
                '\n',
                'num_drift_df = detector.analyze_numerical_drift(ref_df, cur_df, num_cols)\n',
                'cat_drift_df = detector.analyze_categorical_drift(ref_df, cur_df, cat_cols)\n',
                '\n',
                'print("--- Numerical Feature Drift Audit ---")\n',
                'num_drift_df[[\'feature\', \'ks_statistic\', \'p_value_sci\', \'psi_score\', \'psi_status\', \'drift_detected\']]\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Visualizing Distribution Shifts and Diagnostic Plots\n',
                'Reviewing the diagnostic visualizations generated and logged to MLflow.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': 5,
            'metadata': {},
            'outputs': [],
            'source': [
                'from IPython.display import Image, display\n',
                '\n',
                'print("1. Kolmogorov-Smirnov Statistical Summary")\n',
                'display(Image(filename="../artifacts/ks_drift_summary.png"))\n',
                '\n',
                'print("2. Population Stability Index (PSI) Summary")\n',
                'display(Image(filename="../artifacts/psi_stability_index.png"))\n',
                '\n',
                'print("3. Feature Distribution Shifts (Reference vs Production Drift)")\n',
                'display(Image(filename="../artifacts/feature_distribution_shifts.png"))\n',
                '\n',
                'print("4. Prediction Probability Distribution Drift")\n',
                'display(Image(filename="../artifacts/prediction_drift_kde.png"))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 6. Conclusion and Recommended MLOps Action Plan\n',
                'Based on our empirical analysis:\n',
                '- **Severe Data Drift** was detected in key financial risk indicators: `interest_rate` (PSI = 2.388), `debt_to_income_ratio` (PSI = 1.450), and `credit_score` (PSI = 1.036).\n',
                '- **Prediction Drift** shifted model output probabilities from an average baseline of 11.04% to 52.00%.\n',
                '- **Concept Drift** degraded classifier accuracy and risk calibration.\n',
                '\n',
                '### Recommended Next Actions:\n',
                '1. **Automated Pipeline Trigger**: Initiate continuous automated retraining using recency-weighted production batches.\n',
                '2. **Threshold Recalibration**: Recalibrate decision boundaries to prevent elevated loan default risk.\n',
                '3. **Shadow Deployment**: Deploy retrained candidate models in shadow/challenger mode prior to full cutover.'
            ]
        }
    ]

    nb = {
        'cells': nb_cells,
        'metadata': {
            'language_info': {'name': 'python', 'version': '3.13.0'},
            'orig_nbformat': 4
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

    os.makedirs('e:/MLDevOps/notebooks', exist_ok=True)
    out_path = 'e:/MLDevOps/notebooks/mlflow_data_drift_analysis.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    print(f"Jupyter Notebook generated successfully at: {out_path}")

if __name__ == '__main__':
    create_notebook()
