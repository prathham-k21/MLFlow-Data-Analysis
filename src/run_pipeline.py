"""
Master MLOps Pipeline Orchestrator:
1. Generates FinTech Reference and Production Data
2. Trains Baseline Model and Logs Experiments to MLflow
3. Detects Covariate Shift (Data Drift), Prediction Drift, and Concept Drift
4. Generates Diagnostic Visualizations and MLflow Artifacts
5. Exports Drift Analysis Results for Assignment Documentation
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.generate_data import save_datasets
from src.train import train_baseline_model
from src.drift_detector import StatisticalDriftDetector

def run_mlops_pipeline(base_dir: str = 'e:/MLDevOps'):
    print("=" * 75)
    print("[INIT] INITIATING MLOPS MLFLOW & DATA DRIFT DETECTION PIPELINE")
    print("   Course: ML DevOps | Assignment: Theory-ISE 2")
    print("   Student: Pratham Kadam | Roll: DS 14 | PRN: 2324000362")
    print("   Domain: Financial Services / FinTech Credit Default Risk")
    print("=" * 75)

    data_dir = os.path.join(base_dir, 'data')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    mlruns_dir = os.path.join(base_dir, 'mlruns')
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(mlruns_dir, exist_ok=True)

    # Step 1: Generate Datasets
    print("\n[Step 1/5] Generating FinTech Reference and Drifted Production Datasets...")
    save_datasets(output_dir=data_dir)
    ref_path = os.path.join(data_dir, 'reference_data.csv')
    cur_path = os.path.join(data_dir, 'current_data.csv')
    
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)

    # Step 2: Configure MLflow Tracking
    print("\n[Step 2/5] Configuring MLflow Experiment Tracking...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = os.path.join(base_dir, 'mlflow.db').replace('\\', '/')
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment_name = "Credit_Default_Risk_Drift_Monitoring"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"MLflow Experiment: '{experiment_name}' (ID: {experiment.experiment_id})")

    # Step 3: Train Baseline Model
    print("\n[Step 3/5] Training Baseline Classifier on Reference Data...")
    train_results = train_baseline_model(data_path=ref_path, artifacts_dir=artifacts_dir)
    pipeline = train_results['pipeline']
    baseline_metrics = train_results['metrics']
    num_cols, cat_cols = train_results['feature_names']
    X_test_ref, y_test_ref, ref_test_probs = train_results['test_set']

    # Step 4: Evaluate on Current Production Data (Concept Drift Analysis)
    print("\n[Step 4/5] Evaluating Model on Post-Deployment Production Data...")
    X_cur = cur_df.drop(columns=['loan_default'])
    y_cur = cur_df['loan_default']
    
    cur_preds = pipeline.predict(X_cur)
    cur_probs = pipeline.predict_proba(X_cur)[:, 1]
    
    # Calculate Production Metrics
    prod_accuracy = round(float(accuracy_score(y_cur, cur_preds)), 4)
    prod_precision = round(float(precision_score(y_cur, cur_preds, zero_division=0)), 4)
    prod_recall = round(float(recall_score(y_cur, cur_preds)), 4)
    prod_f1 = round(float(f1_score(y_cur, cur_preds)), 4)
    prod_roc_auc = round(float(roc_auc_score(y_cur, cur_probs)), 4)
    
    auc_drop_pct = round(((baseline_metrics['baseline_roc_auc'] - prod_roc_auc) / baseline_metrics['baseline_roc_auc']) * 100, 2)
    f1_drop_pct = round(((baseline_metrics['baseline_f1_score'] - prod_f1) / baseline_metrics['baseline_f1_score']) * 100, 2)

    print(f"Production Performance Degradation:")
    print(f"  • Baseline ROC-AUC: {baseline_metrics['baseline_roc_auc']:.4f}  -->  Production ROC-AUC: {prod_roc_auc:.4f} (Drop: {auc_drop_pct}%)")
    print(f"  • Baseline F1-Score: {baseline_metrics['baseline_f1_score']:.4f}  -->  Production F1-Score: {prod_f1:.4f} (Drop: {f1_drop_pct}%)")

    # Step 5: Statistical Drift Detection
    print("\n[Step 5/5] Executing Statistical Drift Tests & Diagnostic Visualizations...")
    detector = StatisticalDriftDetector(significance_level=0.05, psi_threshold=0.25)
    
    # Numerical Drift (KS, PSI, Wasserstein)
    num_drift_df = detector.analyze_numerical_drift(ref_df, cur_df, num_cols)
    print("\nNumerical Feature Drift Results:")
    print(num_drift_df[['feature', 'ks_statistic', 'p_value_sci', 'psi_score', 'psi_status', 'drift_detected']].to_string(index=False))

    # Categorical Drift (Chi-Square)
    cat_drift_df = detector.analyze_categorical_drift(ref_df, cur_df, cat_cols)
    print("\nCategorical Feature Drift Results:")
    print(cat_drift_df.to_string(index=False))

    # Prediction Probability Drift
    ref_all_probs = pipeline.predict_proba(ref_df[num_cols + cat_cols])[:, 1]
    cur_all_probs = cur_probs
    pred_drift_dict = detector.analyze_prediction_drift(ref_all_probs, cur_all_probs)
    print(f"\nPrediction Output Drift (KS Stat: {pred_drift_dict['ks_statistic']}, PSI: {pred_drift_dict['psi_score']}, Drift: {pred_drift_dict['prediction_drift_detected']})")

    # Generate Publication Plots
    plot_paths = detector.generate_visualizations(
        ref_df=ref_df, cur_df=cur_df, num_drift_df=num_drift_df,
        ref_probs=ref_all_probs, cur_probs=cur_all_probs, output_dir=artifacts_dir
    )

    # Calculate aggregate summary stats
    num_drifted = int(num_drift_df['drift_detected'].sum())
    total_features = len(num_cols) + len(cat_cols)
    drifted_ratio = round(num_drifted / len(num_cols), 4)
    max_ks = float(num_drift_df['ks_statistic'].max())
    max_psi = float(num_drift_df['psi_score'].max())
    overall_drift_detected = bool(drifted_ratio >= 0.30 or pred_drift_dict['prediction_drift_detected'])

    # Compile Full JSON Report
    report_data = {
        "metadata": {
            "course": "ML DevOps (UAMPC0703)",
            "assignment": "Theory-ISE 2: MLflow Data Drift Analysis",
            "student_name": "Pratham Kadam",
            "roll_number": "DS 14",
            "prn": "2324000362",
            "class_branch": "CSE (Data Science) Final Year - Sem VII",
            "academic_year": "2026-27",
            "github_repository": "https://github.com/prathham-k21/MLFlow-Data-Analysis",
            "domain": "Financial Services / FinTech Credit Default Risk",
            "model_type": "GradientBoostingClassifier"
        },
        "environment": {
            "os": "Windows 11 (64-bit)",
            "python_version": "3.13.0",
            "mlflow_version": mlflow.__version__
        },
        "baseline_metrics": baseline_metrics,
        "production_metrics": {
            "production_accuracy": prod_accuracy,
            "production_precision": prod_precision,
            "production_recall": prod_recall,
            "production_f1_score": prod_f1,
            "production_roc_auc": prod_roc_auc,
            "roc_auc_drop_pct": auc_drop_pct,
            "f1_score_drop_pct": f1_drop_pct
        },
        "drift_summary": {
            "total_numerical_features": len(num_cols),
            "drifted_numerical_features": num_drifted,
            "drifted_features_ratio": drifted_ratio,
            "max_ks_statistic": max_ks,
            "max_psi_score": max_psi,
            "prediction_drift": pred_drift_dict,
            "overall_data_drift_detected": overall_drift_detected,
            "concept_drift_detected": bool(auc_drop_pct > 5.0)
        },
        "numerical_drift_table": num_drift_df.to_dict(orient='records'),
        "categorical_drift_table": cat_drift_df.to_dict(orient='records')
    }

    report_path = os.path.join(artifacts_dir, 'drift_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    # MLflow Run Logging
    print("\n[MLflow Logging] Recording Run into MLflow Experiment...")
    with mlflow.start_run(run_name="GradientBoosting_Baseline_and_Drift_Audit") as run:
        run_id = run.info.run_id
        print(f"Active MLflow Run ID: {run_id}")
        
        # Tags
        mlflow.set_tag("student_name", "Pratham Kadam")
        mlflow.set_tag("roll_number", "DS 14")
        mlflow.set_tag("prn", "2324000362")
        mlflow.set_tag("domain", "FinTech_Credit_Default")
        mlflow.set_tag("drift_detected", str(overall_drift_detected))

        # Parameters
        params = {
            "model_type": "GradientBoostingClassifier",
            "n_estimators": 120,
            "learning_rate": 0.08,
            "max_depth": 4,
            "subsample": 0.85,
            "ks_alpha_threshold": 0.05,
            "psi_critical_threshold": 0.25,
            "reference_sample_size": len(ref_df),
            "current_sample_size": len(cur_df),
            "random_seed": 42
        }
        mlflow.log_params(params)

        # Metrics (Baseline + Production + Drift)
        all_metrics = {
            **baseline_metrics,
            "prod_accuracy": prod_accuracy,
            "prod_precision": prod_precision,
            "prod_recall": prod_recall,
            "prod_f1_score": prod_f1,
            "prod_roc_auc": prod_roc_auc,
            "roc_auc_drop_pct": auc_drop_pct,
            "f1_score_drop_pct": f1_drop_pct,
            "num_drifted_features": num_drifted,
            "drifted_features_ratio": drifted_ratio,
            "max_ks_statistic": max_ks,
            "max_psi_score": max_psi,
            "pred_drift_ks_stat": pred_drift_dict['ks_statistic'],
            "pred_drift_psi": pred_drift_dict['psi_score']
        }
        mlflow.log_metrics(all_metrics)

        # Artifacts
        for art_name, art_file in train_results['artifacts'].items():
            mlflow.log_artifact(art_file, artifact_path="model_evaluation")
            
        for art_name, art_file in plot_paths.items():
            mlflow.log_artifact(art_file, artifact_path="drift_diagnostics")
            
        mlflow.log_artifact(report_path, artifact_path="audit_reports")

        # Model Logging with Signature
        sample_input = X_test_ref.head(5)
        signature = infer_signature(sample_input, pipeline.predict(sample_input))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="credit_risk_model",
            signature=signature,
            input_example=sample_input
        )
        print(f"MLflow Run successfully logged with all metrics, artifacts, and serialized model.")

    print("\n" + "=" * 75)
    print("[SUCCESS] MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   MLflow Tracking URI: file:///{mlruns_dir.replace('\\', '/')}")
    print(f"   Run ID: {run_id}")
    print(f"   Overall Data Drift Detected: {overall_drift_detected}")
    print(f"   Drifted Numerical Features: {num_drifted}/{len(num_cols)} ({drifted_ratio:.1%})")
    print(f"   Performance Degradation: ROC-AUC dropped from {baseline_metrics['baseline_roc_auc']:.4f} to {prod_roc_auc:.4f}")
    print("=" * 75)
    return report_data

if __name__ == '__main__':
    run_mlops_pipeline()
