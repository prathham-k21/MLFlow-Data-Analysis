"""
Populates Theory-ISE2-MLFlow Data Analysis.docx with complete, professional
technical content, formatted statistical tables, embedded diagnostic figures,
and converts the output to Theory-ISE2-MLFlow Data Analysis.pdf via Microsoft Word COM.
"""

import os
import shutil
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def prevent_row_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

def set_repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

def populate_word_document():
    base_dir = 'e:/MLDevOps'
    doc_path = os.path.join(base_dir, 'Theory-ISE2-MLFlow Data Analysis.docx')
    backup_path = os.path.join(base_dir, 'Theory-ISE2-MLFlow Data Analysis_Template_Backup.docx')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    
    # 1. Ensure backup exists
    if not os.path.exists(backup_path):
        shutil.copyfile(doc_path, backup_path)
        print(f"Original template backed up to: {backup_path}")

    with open(os.path.join(artifacts_dir, 'drift_analysis_report.json'), 'r') as f:
        report_data = json.load(f)

    doc = docx.Document(backup_path)

    # 2. Fill Details Table (Table 0)
    table = doc.tables[0]
    details_data = [
        ("Student Name", "Pratham Kadam"),
        ("Roll Number(s)", "DS 14 (PRN: 2324000362)"),
        ("Department / Class", "Final Year B.Tech CSE (Data Science) - Semester VII"),
        ("Academic Year", "A.Y 2026-27"),
        ("GitHub Repository URL", "https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis")
    ]
    
    for i, (label, val) in enumerate(details_data, start=1):
        cell_val = table.rows[i].cells[1]
        cell_val.text = ""
        p = cell_val.paragraphs[0]
        run = p.add_run(val)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0, 51, 102)

    # Helper functions
    def append_answer(paragraph, answer_text, is_bold=False):
        run = paragraph.add_run(" " + answer_text)
        run.bold = is_bold
        run.font.name = 'Arial'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(35, 35, 35)

    def set_content(paragraph, new_text):
        paragraph.text = new_text
        for run in paragraph.runs:
            run.font.name = 'Arial'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(35, 35, 35)

    # 3. Fill in Paragraphs
    p_interp = None
    for p in doc.paragraphs:
        txt = p.text.strip()
        
        # 1.1 Environment Setup
        if txt == "Operating System:":
            append_answer(p, "Windows 11 Enterprise (64-bit Edition)", is_bold=True)
        elif txt == "Python Version:":
            append_answer(p, "Python 3.13.0 (64-bit AMD64)", is_bold=True)
        elif txt == "MLflow Version:":
            append_answer(p, f"{report_data['environment']['mlflow_version']} (Modern SQLite tracking backend)", is_bold=True)
        elif txt == "Installation and verification status:":
            append_answer(p, "Verified and Operational. Installed via 'pip install mlflow'. Tracking backend initialized using SQLite ('sqlite:///e:/MLDevOps/mlflow.db') with MLflow UI accessible at http://127.0.0.1:5000.")

        # 1.2 Experiment Tracking
        elif txt == "Experiment Name:":
            append_answer(p, "Credit_Default_Risk_Drift_Monitoring (Experiment ID: 1)", is_bold=True)
        elif txt == "Experiment Objective:":
            append_answer(p, "Track baseline Gradient Boosting model training, hyperparameter optimization, model versioning, evaluation metrics (ROC-AUC, Precision, Recall, F1, Accuracy, Brier Loss), and statistical data drift diagnostics across baseline reference data and production batches.")
        elif txt == "Parameters, metrics or artifacts recorded:":
            append_answer(p, "• Parameters: model_type=GradientBoostingClassifier, n_estimators=120, learning_rate=0.08, max_depth=4, subsample=0.85, random_seed=42, ks_alpha_threshold=0.05, psi_critical_threshold=0.25.\n"
                             "• Baseline Metrics: accuracy=0.8910, roc_auc=0.7853, precision=0.5789, recall=0.0982, f1_score=0.1679, brier_score=0.0856.\n"
                             "• Production & Drift Metrics: prod_accuracy=0.6930, prod_roc_auc=0.7730, max_ks=0.6234, max_psi=2.3880, pred_ks=0.6654, pred_psi=2.7359.\n"
                             "• Artifacts Recorded: Scikit-Learn Pipeline model with signature & input example, confusion_matrix.png, roc_curve.png, feature_importance.png, ks_drift_summary.png, psi_stability_index.png, feature_distribution_shifts.png, prediction_drift_kde.png, and drift_analysis_report.json.")
        elif txt == "Brief description of successful execution:":
            append_answer(p, "Pipeline execution automated via 'src/run_pipeline.py'. An MLflow Run titled 'GradientBoosting_Baseline_and_Drift_Audit' was executed. Training parameters, classification metrics, statistical drift p-values, PSI scores, diagnostic charts, and the complete serialized Scikit-Learn model pipeline were logged to MLflow without error.")

        # 2. Dataset Identification
        elif txt == "Briefly describe the problem addressed in the selected internship domain.":
            set_content(p, "In the Financial Services and FinTech domain, consumer credit institutions face substantial portfolio risk from borrower defaults. Underwriting models predict loan default risk based on borrower leverage, credit history, and income. However, real-world macroeconomic volatility—such as inflation, rising policy interest rates, and escalating cost-of-living—induces severe distribution shifts over time. When borrower debt burdens increase and disposable income contracts, models trained on historical data experience both Data Drift (Covariate Shift) and Concept Drift. This project establishes an automated MLOps drift monitoring pipeline to audit credit default models in real time, detecting distribution divergence before financial losses occur.")

        elif txt == "Dataset Name:":
            append_answer(p, "FinTech Lending & Credit Default Risk Dataset", is_bold=True)
        elif txt == "Dataset Source:":
            append_answer(p, "Synthesized from industry-standard credit risk distributions aligned with LendingClub, German Credit, and Home Credit Default Risk empirical benchmarks.")
        elif txt == "Number of Records:":
            append_answer(p, "10,000 total records (5,000 Reference Baseline loans + 5,000 Production Drift loans)", is_bold=True)
        elif txt == "Important Features:":
            append_answer(p, "annual_income, debt_to_income_ratio, credit_score, loan_amount, interest_rate, employment_length_years, revolving_balance, num_derogatory_records, home_ownership, loan_purpose.")
        elif txt == "Target Variable (if applicable):":
            append_answer(p, "loan_default (Binary: 0 = Non-Default / Fully Paid, 1 = Default / Delinquent)", is_bold=True)
        elif txt.startswith("Use Case Category:"):
            set_content(p, "Use Case Category: Supervised Machine Learning (Binary Classification with Automated Distribution Shift & Data Drift Monitoring)")

        elif txt == "Explain why the dataset is relevant to the selected problem.":
            set_content(p, "Credit scoring models are uniquely vulnerable to economic cycle fluctuations. In consumer lending, debt-to-income ratio (DTI), credit bureau score (FICO), annual income, and interest rates serve as core risk determinants. Under macroeconomic stress, rising interest rates and inflation naturally shift feature distributions (causing Covariate Shift) while elevating default rates across previously medium-risk tiers (causing Concept Drift). This dataset mirrors real-world FinTech lending dynamics, making it an ideal benchmark for demonstrating MLflow tracking, statistical hypothesis testing (KS-test), and regulatory Population Stability Index (PSI) audits.")

        # 3. Drift Scenario and Detection Method
        elif txt == "Identify the relevant drift type: Data Drift / Concept Drift / Prediction Drift.":
            set_content(p, "The pipeline systematically detects and audits all three interrelated drift phenomena:\n"
                           "1. Data Drift (Covariate Shift): Marked shift in the marginal input feature distributions P(X) between reference baseline and production data. In our audit, 100% of numerical features (8/8) exhibited significant drift (p < 0.05), with severe shifts in interest_rate (PSI = 2.388), debt_to_income_ratio (PSI = 1.450), and credit_score (PSI = 1.036).\n"
                           "2. Concept Drift: Alteration in the conditional relationship P(Y|X) between input applicant features and loan default probabilities under macroeconomic stress, causing baseline model accuracy to drop from 89.10% to 69.30% and ROC-AUC from 0.7853 to 0.7730.\n"
                           "3. Prediction Drift: Severe rightward distribution shift in model output probabilities P(Y_hat), where the mean predicted default probability surged from 11.04% at baseline to 52.00% in production (KS = 0.6654, p < 1e-300, PSI = 2.7359).")

        elif txt == "Briefly describe the datasets or model outputs compared, including whether the drift is naturally observed or simulated.":
            set_content(p, "The drift evaluation compared two datasets representing distinct operational periods:\n"
                           "• Reference Data (Baseline / Training): 5,000 historical loan applications processed during stable macroeconomic conditions (Months 1–6), featuring an 11.16% baseline default rate.\n"
                           "• Current Data (Production / Inference): 5,000 post-deployment loan applications collected during an inflationary economic contraction (Months 7–12), where average interest rates rose from 10.46% to 15.35% (+46.8%), debt-to-income ratios rose from 18.45% to 26.81% (+45.3%), and the observed default rate reached 71.68%.\n"
                           "The drift is naturally simulated according to empirical FinTech macroeconomic cycles.")

        elif txt == "Method / Statistical Test / Tool Used:":
            append_answer(p, "1. Two-Sample Kolmogorov-Smirnov (KS) Test (scipy.stats.ks_2samp) for continuous numerical variables.\n"
                             "2. Population Stability Index (PSI) using 10 deciles against banking regulatory standards.\n"
                             "3. Wasserstein Distance (Earth Mover's Distance) for distribution distance quantification.\n"
                             "4. Chi-Square Goodness-of-Fit Test (scipy.stats.chisquare) for categorical feature shifts.\n"
                             "5. MLflow Tracking Service for automated metric logging, parameter auditing, and artifact registry.")
        elif txt == "Features or outputs analyzed:":
            append_answer(p, "All 8 continuous numerical features (annual_income, debt_to_income_ratio, credit_score, loan_amount, interest_rate, employment_length_years, revolving_balance, num_derogatory_records), 2 categorical features (home_ownership, loan_purpose), and model prediction probabilities (predicted_default_prob).")
        elif txt == "Detection threshold or criteria, if applicable:":
            append_answer(p, "• KS-Test: Significance level alpha = 0.05 (p-value < 0.05 rejects H0, confirming data drift).\n"
                             "• PSI Thresholds: PSI < 0.10 (No significant change / Stable); 0.10 <= PSI < 0.25 (Moderate shift / Close monitoring); PSI >= 0.25 (Severe drift / Action required).\n"
                             "• Dataset-Level Drift Rule: Trigger pipeline alert if >= 30% of features exhibit statistically significant drift or if Output Prediction PSI >= 0.25.")

        # 4. Results, Interpretation and Conclusion
        elif txt == "Summarize the important findings. Detailed screenshots, plots and reports should be maintained in GitHub.":
            set_content(p, "Statistical drift analysis confirmed that 100% of continuous numerical features and both categorical features experienced significant distribution divergence. Severe drift was observed in interest rates (KS=0.6234, PSI=2.388), debt-to-income ratio (KS=0.4978, PSI=1.450), and credit scores (KS=0.4250, PSI=1.036). Model prediction probabilities shifted drastically from a baseline mean of 11.04% to 52.00% (KS=0.6654, PSI=2.7359). Due to concept drift under macroeconomic stress, model accuracy dropped from 89.10% to 69.30%, and ROC-AUC declined from 0.7853 to 0.7730. Table 1 summarizes the empirical numerical drift results:")

        elif txt == "4.2 Interpretation":
            p_interp = p
        elif txt == "Explain what the findings mean for the selected internship domain.":
            set_content(p, "In the FinTech credit lending domain, these empirical findings indicate severe credit deterioration across the loan applicant pool. Borrowers are requesting higher loan principal amounts despite experiencing an 18.65% drop in real disposable income, leading to an alarming 45.3% surge in debt-to-income ratios and higher revolving credit burdens (+69.48%). Because the baseline model was trained under stable economic conditions, its risk calibration is invalid for the current applicant pool. The model underestimates risk at the lower score bands, leading to elevated false approvals and heightened delinquency rates. Continuous drift detection successfully flag these systemic shifts before catastrophic loan defaults materialize.")

        elif txt == "State whether drift was detected, whether the result was inconclusive, or whether no significant drift was found. Mention the appropriate next action.":
            set_content(p, "Conclusion: SEVERE DATA DRIFT AND CONCEPT DRIFT DETECTED.\n"
                           "All statistical hypothesis tests rejected the null hypothesis (p < 0.05), with 5 out of 8 features and model output probabilities far exceeding the critical PSI threshold (>= 0.25).\n\n"
                           "Recommended Engineering & Operational Actions:\n"
                           "1. Automated Model Retraining: Trigger CI/CD retraining via MLflow using recency-weighted production data to adapt to current economic conditions.\n"
                           "2. Decision Threshold Recalibration: Lower classification threshold from 0.50 to 0.35 to account for higher credit default exposure and optimize financial loss matrices.\n"
                           "3. Feature Store Augmentation: Incorporate macroeconomic leading indicators (e.g., inflation index, central bank rate) into the feature store.\n"
                           "4. Shadow Deployment: Deploy retrained model candidate in shadow mode under MLflow Model Registry to benchmark performance against the incumbent before live rollout.")

        # 5. GitHub Repository and Reproducibility
        elif txt == "GitHub Repository URL:":
            append_answer(p, "https://github.com/prathamkadam/MLOps-MLflow-Drift-Analysis", is_bold=True)
        elif txt == "Source code / Notebook:":
            append_answer(p, "Complete modular source code in 'src/' (generate_data.py, train.py, drift_detector.py, run_pipeline.py) and interactive Jupyter Notebook in 'notebooks/mlflow_data_drift_analysis.ipynb'.")
        elif txt == "README and execution instructions:":
            append_answer(p, "Comprehensive README.md with end-to-end architecture diagrams, mathematical formulations, quickstart instructions ('pip install -r requirements.txt', 'python src/run_pipeline.py'), and MLflow UI launch guide ('mlflow ui --backend-store-uri sqlite:///mlflow.db').")
        elif txt == "Requirements / Dependencies:":
            append_answer(p, "mlflow>=3.0.0, scikit-learn>=1.5.0, pandas>=2.0.0, numpy>=1.26.0, scipy>=1.14.0, matplotlib>=3.9.0, seaborn>=0.13.0, python-docx>=1.1.0, joblib>=1.4.0.")

    # 4. Insert Table 1 directly into Section 4.1 (before Section 4.2)
    if p_interp is not None:
        table_num = doc.add_table(rows=1, cols=6)
        table_num.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        headers = ["Feature", "KS Stat (D)", "p-value", "PSI Score", "FinTech PSI Status", "Drift Detected"]
        hdr_row = table_num.rows[0]
        set_repeat_header(hdr_row)
        prevent_row_split(hdr_row)
        for idx, h in enumerate(headers):
            cell = hdr_row.cells[idx]
            cell.text = h
            set_cell_background(cell, "003366")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            p.runs[0].bold = True
            p.runs[0].font.size = Pt(8.5)
            p.runs[0].font.name = 'Arial'

        for item in report_data['numerical_drift_table']:
            row = table_num.add_row()
            prevent_row_split(row)
            row_cells = row.cells
            row_cells[0].text = item['feature']
            row_cells[1].text = f"{item['ks_statistic']:.4f}"
            row_cells[2].text = item['p_value_sci']
            row_cells[3].text = f"{item['psi_score']:.4f}"
            row_cells[4].text = item['psi_status']
            row_cells[5].text = "YES" if item['drift_detected'] else "NO"
            
            # Highlight severe vs moderate
            if "Severe" in item['psi_status']:
                set_cell_background(row_cells[4], "FADBD8")
                set_cell_background(row_cells[5], "FADBD8")
            else:
                set_cell_background(row_cells[4], "FCF3CF")
                set_cell_background(row_cells[5], "FCF3CF")

            for c in row_cells:
                set_cell_margins(c, top=40, bottom=40, left=60, right=60)
                p = c.paragraphs[0]
                p.runs[0].font.size = Pt(8)
                p.runs[0].font.name = 'Arial'

        # Insert table before 4.2
        p_interp._p.addprevious(table_num._tbl)
        
        # Add a caption paragraph before 4.2
        p_cap = doc.add_paragraph("Table 1: Empirical Statistical Drift Detection Summary across All Features")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].bold = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.runs[0].font.color.rgb = RGBColor(0, 51, 102)
        table_num._tbl.addprevious(p_cap._p)

    # 5. Add Section 6: Diagnostic Visualizations & MLflow Artifacts
    p_sec6 = doc.add_paragraph()
    p_sec6.paragraph_format.space_before = Pt(18)
    run_sec6 = p_sec6.add_run("6. Diagnostic Visualizations & MLflow Experiment Artifacts")
    run_sec6.bold = True
    run_sec6.font.size = Pt(12)
    run_sec6.font.name = 'Arial'
    run_sec6.font.color.rgb = RGBColor(0, 51, 102)

    p_sec6_intro = doc.add_paragraph("The figures below represent the high-resolution diagnostic visualizations generated by the statistical drift engine and logged directly into the MLflow Experiment tracking registry:")
    p_sec6_intro.runs[0].font.size = Pt(9.5)
    p_sec6_intro.runs[0].font.name = 'Arial'

    fig_configs = [
        ("Figure 1: Two-Sample Kolmogorov-Smirnov Test Statistics by Feature", os.path.join(artifacts_dir, 'ks_drift_summary.png'), Inches(5.8)),
        ("Figure 2: Population Stability Index (PSI) against FinTech Regulatory Thresholds", os.path.join(artifacts_dir, 'psi_stability_index.png'), Inches(5.8)),
        ("Figure 3: Marginal Feature Distribution Shifts (Covariate Shift Analysis)", os.path.join(artifacts_dir, 'feature_distribution_shifts.png'), Inches(5.8)),
        ("Figure 4: Prediction Probability Output Distribution Drift (Reference vs Production)", os.path.join(artifacts_dir, 'prediction_drift_kde.png'), Inches(5.5)),
        ("Figure 5: Baseline Classifier ROC-AUC Characteristic Curve", os.path.join(artifacts_dir, 'roc_curve.png'), Inches(4.5)),
        ("Figure 6: Baseline Classifier Confusion Matrix", os.path.join(artifacts_dir, 'confusion_matrix.png'), Inches(4.2))
    ]

    for caption, img_path, width in fig_configs:
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(12)
            p_img.paragraph_format.space_after = Pt(4)
            run_img = p_img.add_run()
            run_img.add_picture(img_path, width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(12)
            run_cap = p_cap.add_run(caption)
            run_cap.font.size = Pt(8.5)
            run_cap.font.italic = True
            run_cap.font.name = 'Arial'
            run_cap.font.color.rgb = RGBColor(80, 80, 80)

    # Save populated DOCX
    out_docx = doc_path
    doc.save(out_docx)
    print(f"Populated Word document successfully saved to: {out_docx}")

    # Convert to PDF using Word COM
    try:
        import win32com.client
        out_pdf = os.path.join(base_dir, 'Theory-ISE2-MLFlow Data Analysis.pdf')
        word = win32com.client.Dispatch('Word.Application')
        word.Visible = False
        doc_com = word.Documents.Open(os.path.abspath(out_docx))
        doc_com.SaveAs(os.path.abspath(out_pdf), FileFormat=17) # 17 = wdFormatPDF
        doc_com.Close()
        word.Quit()
        print(f"High-Fidelity PDF successfully exported to: {out_pdf}")
    except Exception as e:
        print(f"PDF export via COM error: {e}")

if __name__ == '__main__':
    populate_word_document()
