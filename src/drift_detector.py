"""
Statistical Drift Detection Engine for MLOps
Implements:
1. Two-sample Kolmogorov-Smirnov (KS) Test for continuous variables
2. Population Stability Index (PSI) with FinTech regulatory thresholds
3. Wasserstein Distance (Earth Mover's Distance)
4. Chi-Square Test for categorical variables
5. Model Prediction Probability Drift & Concept Drift metrics
6. Publication-quality diagnostic visualization generators
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalDriftDetector:
    def __init__(self, significance_level: float = 0.05, psi_threshold: float = 0.25):
        self.alpha = significance_level
        self.psi_threshold = psi_threshold
        
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates the Population Stability Index (PSI) between reference and current samples.
        """
        # Remove NaNs
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        # Calculate quantile bin edges from reference dataset
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(reference, quantiles)
        bin_edges = np.unique(bin_edges) # Handle duplicate edges
        
        if len(bin_edges) < 2:
            return 0.0
            
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Calculate counts per bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Avoid zero division with Laplace smoothing
        ref_pct = (ref_counts + 1e-4) / (len(reference) + 1e-4 * len(ref_counts))
        cur_pct = (cur_counts + 1e-4) / (len(current) + 1e-4 * len(cur_counts))
        
        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi_value)
        
    def analyze_numerical_drift(self, ref_df: pd.DataFrame, cur_df: pd.DataFrame, num_features: list) -> pd.DataFrame:
        results = []
        for feat in num_features:
            ref_vals = ref_df[feat].dropna().values
            cur_vals = cur_df[feat].dropna().values
            
            # 1. Two-Sample Kolmogorov-Smirnov Test
            ks_res = stats.ks_2samp(ref_vals, cur_vals)
            ks_stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)
            
            # 2. Population Stability Index (PSI)
            psi_score = self.calculate_psi(ref_vals, cur_vals)
            
            # 3. Wasserstein Distance
            w_dist = float(stats.wasserstein_distance(ref_vals, cur_vals))
            
            # Mean & Std shifts
            ref_mean = float(np.mean(ref_vals))
            cur_mean = float(np.mean(cur_vals))
            mean_pct_change = float(((cur_mean - ref_mean) / (abs(ref_mean) + 1e-5)) * 100)
            
            # Drift classification
            is_ks_drift = bool(p_val < self.alpha)
            is_psi_drift = bool(psi_score >= self.psi_threshold)
            drift_detected = bool(is_ks_drift or is_psi_drift)
            
            if psi_score < 0.10:
                psi_status = "No Drift (<0.10)"
            elif psi_score < 0.25:
                psi_status = "Moderate Drift (0.10-0.25)"
            else:
                psi_status = "Severe Drift (>=0.25)"
                
            results.append({
                'feature': feat,
                'ks_statistic': round(ks_stat, 4),
                'p_value': p_val,
                'p_value_sci': f"{p_val:.2e}",
                'psi_score': round(psi_score, 4),
                'psi_status': psi_status,
                'wasserstein_distance': round(w_dist, 4),
                'reference_mean': round(ref_mean, 2),
                'current_mean': round(cur_mean, 2),
                'mean_pct_change': round(mean_pct_change, 2),
                'drift_detected': drift_detected
            })
            
        return pd.DataFrame(results)

    def analyze_categorical_drift(self, ref_df: pd.DataFrame, cur_df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
        results = []
        for feat in cat_features:
            ref_counts = ref_df[feat].value_counts(normalize=True)
            cur_counts = cur_df[feat].value_counts(normalize=True)
            
            all_cats = list(set(ref_counts.index).union(set(cur_counts.index)))
            ref_dist = np.array([ref_counts.get(c, 1e-4) for c in all_cats])
            cur_dist = np.array([cur_counts.get(c, 1e-4) for c in all_cats])
            
            # Normalize to 1
            ref_dist = ref_dist / ref_dist.sum()
            cur_dist = cur_dist / cur_dist.sum()
            
            # Chi-square test on expected counts
            expected_counts = ref_dist * len(cur_df)
            observed_counts = cur_dist * len(cur_df)
            chi2_stat, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
            
            results.append({
                'feature': feat,
                'chi2_statistic': round(float(chi2_stat), 4),
                'p_value': float(p_val),
                'p_value_sci': f"{p_val:.2e}",
                'drift_detected': bool(p_val < self.alpha)
            })
        return pd.DataFrame(results)

    def analyze_prediction_drift(self, ref_probs: np.ndarray, cur_probs: np.ndarray) -> dict:
        ks_res = stats.ks_2samp(ref_probs, cur_probs)
        psi_score = self.calculate_psi(ref_probs, cur_probs)
        w_dist = float(stats.wasserstein_distance(ref_probs, cur_probs))
        
        return {
            'ref_mean_prob': round(float(np.mean(ref_probs)), 4),
            'cur_mean_prob': round(float(np.mean(cur_probs)), 4),
            'ks_statistic': round(float(ks_res.statistic), 4),
            'p_value': float(ks_res.pvalue),
            'psi_score': round(psi_score, 4),
            'wasserstein_distance': round(w_dist, 4),
            'prediction_drift_detected': bool(ks_res.pvalue < self.alpha)
        }

    def generate_visualizations(self, ref_df: pd.DataFrame, cur_df: pd.DataFrame, 
                                num_drift_df: pd.DataFrame, ref_probs: np.ndarray, 
                                cur_probs: np.ndarray, output_dir: str = 'e:/MLDevOps/artifacts'):
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font="sans-serif")
        
        # 1. KS-Statistic Summary Bar Chart
        plt.figure(figsize=(10, 5), dpi=300)
        sorted_df = num_drift_df.sort_values(by='ks_statistic', ascending=True)
        colors = ['#e74c3c' if d else '#2ecc71' for d in sorted_df['drift_detected']]
        
        bars = plt.barh(sorted_df['feature'], sorted_df['ks_statistic'], color=colors, edgecolor='black', alpha=0.85)
        plt.axvline(x=0.05, color='#d35400', linestyle='--', linewidth=1.5, label='Statistical Drift Threshold (0.05)')
        plt.title('Two-Sample Kolmogorov-Smirnov Test (Data Drift by Feature)', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('KS Statistic (D-value)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Add labels
        for bar, p_val in zip(bars, sorted_df['p_value_sci']):
            w = bar.get_width()
            plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"p={p_val}", va='center', fontsize=9, fontweight='medium')
            
        plt.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        ks_path = os.path.join(output_dir, 'ks_drift_summary.png')
        plt.savefig(ks_path)
        plt.close()
        
        # 2. Population Stability Index (PSI) Bar Chart
        plt.figure(figsize=(10, 5), dpi=300)
        psi_sorted = num_drift_df.sort_values(by='psi_score', ascending=True)
        psi_colors = []
        for val in psi_sorted['psi_score']:
            if val < 0.10:
                psi_colors.append('#2ecc71')  # Green (Stable)
            elif val < 0.25:
                psi_colors.append('#f39c12')  # Orange (Moderate)
            else:
                psi_colors.append('#e74c3c')  # Red (Severe)
                
        bars = plt.barh(psi_sorted['feature'], psi_sorted['psi_score'], color=psi_colors, edgecolor='black', alpha=0.85)
        plt.axvline(x=0.10, color='#27ae60', linestyle=':', linewidth=1.5, label='Slight Shift (0.10)')
        plt.axvline(x=0.25, color='#c0392b', linestyle='--', linewidth=1.8, label='Action Required Threshold (0.25)')
        plt.title('Population Stability Index (PSI) - FinTech Regulatory Standards', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('PSI Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')
            
        plt.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        psi_path = os.path.join(output_dir, 'psi_stability_index.png')
        plt.savefig(psi_path)
        plt.close()

        # 3. Density / KDE Comparison for Top Drifted Features
        top_features = ['debt_to_income_ratio', 'credit_score', 'interest_rate', 'annual_income']
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
        axes = axes.flatten()
        
        for i, feat in enumerate(top_features):
            ax = axes[i]
            sns.kdeplot(ref_df[feat], ax=ax, label='Reference (Baseline)', color='#2980b9', fill=True, alpha=0.35, linewidth=2)
            sns.kdeplot(cur_df[feat], ax=ax, label='Current (Production Drift)', color='#e74c3c', fill=True, alpha=0.35, linewidth=2)
            
            ks_row = num_drift_df[num_drift_df['feature'] == feat].iloc[0]
            ax.set_title(f"{feat.replace('_', ' ').title()}\n(KS={ks_row['ks_statistic']}, PSI={ks_row['psi_score']}, p={ks_row['p_value_sci']})",
                         fontsize=11, fontweight='bold')
            ax.set_xlabel(feat)
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
            
        plt.suptitle('Marginal Feature Distribution Shifts (Covariate Shift Analysis)', fontsize=15, fontweight='bold', y=0.99)
        plt.tight_layout()
        dist_path = os.path.join(output_dir, 'feature_distribution_shifts.png')
        plt.savefig(dist_path)
        plt.close()

        # 4. Model Prediction Probability Drift (KDE)
        plt.figure(figsize=(9, 5), dpi=300)
        sns.kdeplot(ref_probs, label=f'Reference Model Output (Mean = {np.mean(ref_probs):.3f})', color='#3498db', fill=True, alpha=0.4, linewidth=2)
        sns.kdeplot(cur_probs, label=f'Current Model Output (Mean = {np.mean(cur_probs):.3f})', color='#e67e22', fill=True, alpha=0.4, linewidth=2)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.2, label='Decision Boundary (0.50)')
        plt.title('Prediction Probability Drift (P(Y_hat) Shift Analysis)', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Predicted Default Probability', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        pred_path = os.path.join(output_dir, 'prediction_drift_kde.png')
        plt.savefig(pred_path)
        plt.close()
        
        print("Generated all drift diagnostic visualizations successfully.")
        return {
            'ks_summary': ks_path,
            'psi_summary': psi_path,
            'feature_distributions': dist_path,
            'prediction_drift': pred_path
        }
