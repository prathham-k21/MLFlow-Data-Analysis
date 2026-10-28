"""
MLflow Model Training and Experiment Tracking Module
Trains a credit default risk classification model on reference baseline data,
computes baseline evaluation metrics, and logs hyperparameters, metrics, and
model artifacts into MLflow.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, brier_score_loss
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def build_preprocessing_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessor

def train_baseline_model(data_path: str = 'e:/MLDevOps/data/reference_data.csv',
                         artifacts_dir: str = 'e:/MLDevOps/artifacts'):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load Reference Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['loan_default'])
    y = df['loan_default']
    
    num_cols = [
        'annual_income', 'debt_to_income_ratio', 'credit_score',
        'loan_amount', 'interest_rate', 'employment_length_years',
        'revolving_balance', 'num_derogatory_records'
    ]
    cat_cols = ['home_ownership', 'loan_purpose']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Build Pipeline
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )
    
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    print("Fitting Gradient Boosting Classifier on Baseline Reference Data...")
    clf_pipeline.fit(X_train, y_train)
    
    # Predict on Baseline Test
    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate Baseline Metrics
    metrics = {
        'baseline_accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'baseline_precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'baseline_recall': round(float(recall_score(y_test, y_pred)), 4),
        'baseline_f1_score': round(float(f1_score(y_test, y_pred)), 4),
        'baseline_roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
        'baseline_brier_score': round(float(brier_score_loss(y_test, y_prob)), 4)
    }
    print(f"Baseline Evaluation: {metrics}")
    
    # Generate Baseline Artifacts
    # A. Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=300)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Paid (0)', 'Default (1)'],
                yticklabels=['Paid (0)', 'Default (1)'])
    plt.title('Baseline Test Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # B. ROC Curve
    plt.figure(figsize=(6, 5), dpi=300)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='#2980b9', lw=2.5, label=f"ROC Curve (AUC = {metrics['baseline_roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Baseline ROC-AUC Characteristic Curve', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(artifacts_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    
    # C. Feature Importance
    plt.figure(figsize=(8, 5), dpi=300)
    # Extract feature names after OneHotEncoding
    ohe_cols = list(clf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + ohe_cols
    importances = clf_pipeline.named_steps['classifier'].feature_importances_
    
    feat_imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(10)
    
    sns.barplot(data=feat_imp_df, x='importance', y='feature', palette='viridis', edgecolor='black')
    plt.title('Top 10 Feature Importances (Gradient Boosting)', fontsize=12, fontweight='bold')
    plt.xlabel('Normalized Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    imp_path = os.path.join(artifacts_dir, 'feature_importance.png')
    plt.savefig(imp_path)
    plt.close()
    
    return {
        'pipeline': clf_pipeline,
        'metrics': metrics,
        'artifacts': {
            'confusion_matrix': cm_path,
            'roc_curve': roc_path,
            'feature_importance': imp_path
        },
        'test_set': (X_test, y_test, y_prob),
        'feature_names': (num_cols, cat_cols)
    }
