"""
Exploratory Data Analysis (EDA) Script
Author: Data Science & HR Analytics Team
Purpose: Conducts deep exploratory analysis on employee behavioral signals,
         analyzes class balances, feature correlations, and mutual information scores.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

def run_eda(data_path="data/employee_features.csv", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("=" * 60)
    print(f"Total Records     : {df.shape[0]}")
    print(f"Total Features    : {df.shape[1]}")
    print(f"Missing Values    : {df.isnull().sum().sum()}")
    print("\nDataset Shape and Data Types:")
    print(df.dtypes.value_counts())
    
    print("\n--- Target Distribution ---")
    print(df['perf_band_next'].value_counts(normalize=True))
    
    # Mutual Information Ranking
    num_cols = df.select_dtypes(include='number').columns
    X_num = df[num_cols]
    y = df['perf_band_next']
    
    mi = mutual_info_classif(X_num, y, random_state=42)
    mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=False)
    
    print("\n--- Top Numeric Features by Mutual Information ---")
    print(mi_series.head(10).round(4))
    
    # Save correlation heatmap
    plt.figure(figsize=(12, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, cmap='vlag', center=0, annot=False)
    plt.title("Correlation Matrix of Employee Metrics", fontsize=14, pad=12)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"\nSaved correlation heatmap to: {corr_path}")
    print("=" * 60)

if __name__ == '__main__':
    run_eda()
