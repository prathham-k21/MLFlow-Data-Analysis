# MLOps MLflow Experiment Tracking and Data Drift Detection Pipeline

**Course:** ML DevOps (Course Code: `UAMPC0703`)  
**Assignment:** Theory-ISE 2  
**Student Name:** Pratham Kadam  
**Roll Number:** DS 14  
**PRN:** 2324000362  
**Class & Branch:** Final Year B.Tech CSE (Data Science) - Semester VII  
**Academic Year:** 2026-27  
**GitHub Repository:** [https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis](https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis)

---

## 📌 Executive Summary & Problem Statement

Financial lending institutions rely heavily on predictive machine learning models to assess borrower creditworthiness and loan default probabilities. In real-world FinTech environments, however, model performance inevitably degrades over time due to **Data Drift (Covariate Shift)** and **Concept Drift** caused by macroeconomic volatility, inflation, interest rate hikes, and evolving borrower risk profiles.

This repository implements a production-grade MLOps pipeline using **MLflow** for experiment tracking and model registry, coupled with an end-to-end **Statistical Drift Detection Engine** incorporating:
1. **Two-Sample Kolmogorov-Smirnov (KS) Hypothesis Testing**
2. **Population Stability Index (PSI)** under Basel / FinTech regulatory standards
3. **Wasserstein Distance (Earth Mover's Distance)**
4. **Chi-Square Goodness-of-Fit Tests** for categorical shifts
5. **Prediction Probability Drift & Concept Drift Evaluation**

---

## 🏗️ Architecture & Pipeline Flow

```
+-----------------------------------------------------------------------------------+
|                           FinTech Lending Data Pipeline                           |
+-----------------------------------------------------------------------------------+
           |                                                |
           v                                                v
[Reference Dataset: 5,000 Loans]                [Current Production: 5,000 Loans]
  - Stable Macro Conditions                       - Economic Stress & Rate Hikes
  - Baseline Default: ~11.16%                     - High DTI, Lower Credit Scores
           |                                                |
           v                                                |
+--------------------------------------+                    |
| Scikit-Learn Pipeline + MLflow       |                    |
|  - StandardScaler + OneHotEncoder   |                    |
|  - GradientBoostingClassifier        |                    |
|  - Hyperparameter & Metric Logging   |                    |
|  - Model Artifact & Signature Export |                    |
+--------------------------------------+                    |
           |                                                |
           v                                                v
+-----------------------------------------------------------------------------------+
|                        Statistical Drift Detection Engine                         |
|  - Kolmogorov-Smirnov (KS) Test (continuous numerical features)                   |
|  - Population Stability Index (PSI) with FinTech thresholds (<0.1, 0.1-0.25, >0.25)|
|  - Chi-Square Goodness-of-Fit (categorical features)                             |
|  - Model Output Probability Drift (KS & PSI on P(Y_hat))                          |
|  - Performance Degradation Audit (ROC-AUC & F1-Score Drops)                       |
+-----------------------------------------------------------------------------------+
           |
           v
+-----------------------------------------------------------------------------------+
|                        MLflow Tracking & Artifact Storage                         |
|  - SQLite DB Backend: sqlite:///mlflow.db                                         |
|  - Visual Artifacts: ks_drift_summary.png, psi_stability_index.png, etc.          |
|  - Audit Report: drift_analysis_report.json                                       |
+-----------------------------------------------------------------------------------+
```

---

## 📊 Statistical Drift Detection Methodology

### 1. Two-Sample Kolmogorov-Smirnov (KS) Test
The two-sample KS test compares the empirical cumulative distribution functions (ECDFs) of reference ($F_{ref}$) and current ($F_{cur}$) datasets:
$$D = \sup_x |F_{ref}(x) - F_{cur}(x)|$$
- **Null Hypothesis ($H_0$):** Both samples originate from identical continuous distributions.
- **Decision Rule:** Reject $H_0$ if $p\text{-value} < \alpha$ (threshold $\alpha = 0.05$).

### 2. Population Stability Index (PSI)
The standard metric utilized across banking risk models to assess shifts in applicant distributions:
$$PSI = \sum_{i=1}^k \left(Actual\%_i - Expected\%_i\right) \times \ln\left(\frac{Actual\%_i}{Expected\%_i}\right)$$
- **$PSI < 0.10$**: Negligible shift; model is stable.
- **$0.10 \le PSI < 0.25$**: Moderate shift; increase monitoring frequency.
- **$PSI \ge 0.25$**: Significant population shift; immediate retraining and review required.

---

## 📈 Empirical Results Summary

### 1. Baseline Model Performance (Reference Test Set)
- **Accuracy:** `89.10%`
- **ROC-AUC Score:** `0.7853`
- **Precision:** `57.89%`
- **Recall:** `9.82%`
- **F1-Score:** `0.1679`
- **Brier Score Loss:** `0.0856`

### 2. Numerical Feature Drift Summary
| Feature | KS Statistic ($D$) | $p$-value | PSI Score | FinTech PSI Status | Drift Detected |
|---|---|---|---|---|---|
| `interest_rate` | **0.6234** | $0.00\times 10^0$ | **2.3880** | **Severe Drift ($\ge 0.25$)** | **YES** |
| `debt_to_income_ratio` | **0.4978** | $0.00\times 10^0$ | **1.4500** | **Severe Drift ($\ge 0.25$)** | **YES** |
| `credit_score` | **0.4250** | $0.00\times 10^0$ | **1.0362** | **Severe Drift ($\ge 0.25$)** | **YES** |
| `loan_amount` | **0.2858** | $2.84\times 10^{-180}$ | **0.4627** | **Severe Drift ($\ge 0.25$)** | **YES** |
| `employment_length_years`| **0.2126** | $2.57\times 10^{-99}$ | **0.2910** | **Severe Drift ($\ge 0.25$)** | **YES** |
| `annual_income` | **0.2084** | $2.03\times 10^{-95}$ | **0.2496** | Moderate Drift ($0.10-0.25$) | **YES** |
| `revolving_balance` | **0.1932** | $5.55\times 10^{-82}$ | **0.2405** | Moderate Drift ($0.10-0.25$) | **YES** |
| `num_derogatory_records`| **0.2210** | $2.37\times 10^{-107}$| **0.2078** | Moderate Drift ($0.10-0.25$) | **YES** |

### 3. Categorical Feature Drift Summary
| Feature | Chi-Square ($\chi^2$) | $p$-value | Drift Detected |
|---|---|---|---|
| `home_ownership` | `400.31` | $1.19\times 10^{-87}$ | **YES** |
| `loan_purpose` | `239.44` | $1.23\times 10^{-50}$ | **YES** |

### 4. Prediction Probability Output Drift
- **Reference Mean Probability:** `11.04%`
- **Production Mean Probability:** `52.00%`
- **Output KS Statistic:** `0.6654` ($p < 10^{-300}$)
- **Output PSI:** `2.7359` (Critical Shift Detected)

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:
```bash
git clone https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis.git
cd MLOps-MLflow-Drift-Analysis
pip install -r requirements.txt
```

### 2. Run the End-to-End MLOps Pipeline
Execute the master orchestration script:
```bash
python src/run_pipeline.py
```
This single command automatically:
1. Synthesizes 5,000 reference and 5,000 production records in `data/`.
2. Connects to the MLflow SQLite backend (`sqlite:///mlflow.db`).
3. Trains and logs the Gradient Boosting baseline pipeline.
4. Performs statistical drift hypothesis tests and PSI audits.
5. Saves high-resolution diagnostic plots in `artifacts/`.
6. Logs metrics, model artifacts, signatures, and reports to MLflow.

### 3. Launch MLflow Tracking UI
To inspect the experiments, metrics, run parameters, and artifact plots interactively:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📁 Repository Directory Structure

```
MLOps-MLflow-Drift-Analysis/
├── artifacts/                           # Diagnostic plots and audit JSON
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── ks_drift_summary.png
│   ├── psi_stability_index.png
│   ├── feature_distribution_shifts.png
│   ├── prediction_drift_kde.png
│   └── drift_analysis_report.json
├── data/                                # Reference & production datasets
│   ├── reference_data.csv
│   ├── current_data.csv
│   └── dataset_metadata.json
├── notebooks/                           # Interactive Jupyter Notebook
│   └── mlflow_data_drift_analysis.ipynb
├── src/                                 # Production Python source modules
│   ├── __init__.py
│   ├── generate_data.py
│   ├── train.py
│   ├── drift_detector.py
│   └── run_pipeline.py
├── scripts/
│   └── make_notebook.py
├── Theory-ISE2-MLFlow Data Analysis.docx # Formatted submission Word document
├── Theory-ISE2-MLFlow Data Analysis.pdf  # Final submission PDF for ERP
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 💡 MLOps Engineering Decisions & Mitigation Plan
1. **Automated Retraining Trigger:** When $\text{Drifted Features Ratio} \ge 30\%$ or $\text{Max PSI} \ge 0.25$, an automated webhook triggers a retraining pipeline using recency-weighted production data.
2. **Dynamic Decision Threshold Calibration:** Due to increased borrower leverage, the fixed 0.50 threshold is adjusted using cost-matrix optimization to maintain strict portfolio delinquency limits.
3. **Challenger Model Shadow Deployment:** The retrained model is deployed in parallel with the incumbent model to evaluate real-world calibration before full routing cutover.
