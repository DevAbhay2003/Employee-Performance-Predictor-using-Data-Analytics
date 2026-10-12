"""
Model Evaluation and Explainability Module
Author: Data Science & HR Analytics Team
Purpose: Generates classification reports, confusion matrices, feature importance plots,
         and conducts HR fairness/bias audits across protected groups.
"""

import os
import shutil
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from src.preprocessing import TARGET_COL

def run_evaluation(model_path="models/employee_perf_model.pkl",
                   test_data_path="data/test_holdout.csv",
                   metadata_path="models/feature_metadata.json",
                   outputs_dir="outputs",
                   images_dir="images"):

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 3: Model Evaluation & Explainability Analysis")
    print("=" * 60)

    # Load model and test data
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Core Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred, digits=4)

    print(f"\nHoldout Test Accuracy    : {acc * 100:.2f}%")
    print(f"Holdout Test Macro F1-Score : {f1_macro:.4f}")
    print(f"Holdout Test Weighted F1   : {f1_weighted:.4f}")
    print("\nDetailed Classification Report:")
    print(report_str)

    # 1. Confusion Matrix Plot
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Employee Performance Prediction - Confusion Matrix', fontsize=13, pad=12)
    plt.xlabel('Predicted Rating Band', fontsize=11)
    plt.ylabel('Actual Rating Band', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    shutil.copy(cm_path, os.path.join(images_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    # 2. Feature Importance Plot
    rf_clf = model.named_steps['classifier']
    feature_names = metadata['features']
    importances = rf_clf.feature_importances_

    # Create top 15 features series
    fi_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=fi_series.values, y=fi_series.index, hue=fi_series.index, palette='crest_r', legend=False)
    plt.title('Top 15 Drivers of Employee Performance (Random Forest)', fontsize=13, pad=12)
    plt.xlabel('Gini Feature Importance', fontsize=11)
    plt.ylabel('Feature Signal', fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(outputs_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    shutil.copy(fi_path, os.path.join(images_dir, "feature_importance.png"))
    plt.close()
    print(f"Feature importance plot saved to: {fi_path}")

    # 3. Performance Band Distribution Plot
    plt.figure(figsize=(7, 4.5))
    df_plot = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_plot_melt = df_plot.melt(var_name='Type', value_name='Band')
    sns.countplot(data=df_plot_melt, x='Band', hue='Type', palette='Set2', order=['Low', 'Medium', 'High'])
    plt.title('Actual vs Predicted Rating Band Distribution', fontsize=13, pad=12)
    plt.xlabel('Performance Band', fontsize=11)
    plt.ylabel('Employee Count', fontsize=11)
    plt.tight_layout()
    dist_path = os.path.join(outputs_dir, "performance_distribution.png")
    plt.savefig(dist_path, dpi=300)
    shutil.copy(dist_path, os.path.join(images_dir, "performance_distribution.png"))
    plt.close()
    print(f"Distribution plot saved to: {dist_path}")

    # 4. Fairness and Bias Audit (Demographic Parity Check)
    print("\n" + "=" * 60)
    print("STEP 4: HR Ethical Fairness & Bias Audit (Gender)")
    print("=" * 60)
    test_with_pred = test_df.copy()
    test_with_pred['predicted_band'] = y_pred

    fairness_records = []
    for g in test_with_pred['gender'].unique():
        sub = test_with_pred[test_with_pred['gender'] == g]
        high_selection_rate = (sub['predicted_band'] == 'High').mean()
        group_acc = accuracy_score(sub[TARGET_COL], sub['predicted_band'])
        fairness_records.append({
            'Gender Group': g,
            'Sample Size': len(sub),
            'High Performer Selection Rate': f"{high_selection_rate * 100:.1f}%",
            'Model Accuracy': f"{group_acc * 100:.1f}%"
        })
    fairness_df = pd.DataFrame(fairness_records)
    print(fairness_df.to_string(index=False))

    # Save Comprehensive Text Report
    report_file = os.path.join(outputs_dir, "metrics_report.txt")
    with open(report_file, 'w') as f:
        f.write("EMPLOYEE PERFORMANCE PREDICTOR - MODEL EVALUATION REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Model Algorithm       : {metadata['model_type']}\n")
        f.write(f"Holdout Test Accuracy : {acc * 100:.2f}%\n")
        f.write(f"Macro F1-Score        : {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score     : {f1_weighted:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_str + "\n\n")
        f.write("TOP 10 FEATURE DRIVERS:\n")
        for rank, (feat, val) in enumerate(fi_series.head(10).items(), 1):
            f.write(f"{rank:2d}. {feat:<30} : {val:.4f}\n")
        f.write("\nFAIRNESS AND BIAS AUDIT:\n")
        f.write(fairness_df.to_string(index=False) + "\n")

    print(f"\nComprehensive metrics report generated at: {report_file}")
    print("=" * 60)

    return {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'report_dict': report_dict,
        'fairness_df': fairness_df
    }

if __name__ == '__main__':
    run_evaluation()
