"""
Model Training Module
Author: Data Science & HR Analytics Team
Purpose: Compares baseline vs. ensemble classifier with 5-fold Stratified CV,
         handles class weighting, and serializes production pipeline artifacts.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, TARGET_COL, ID_COL, get_feature_lists

def run_training(data_path="data/employee_features.csv", models_dir="models", outputs_dir="outputs"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading Dataset for Model Training")
    print("=" * 60)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data/generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}")

    # Stratified Train-Test Split (80% Train, 20% Holdout Test)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    test_save_path = "data/test_holdout.csv"
    test_df.to_csv(test_save_path, index=False)
    print(f"Saved holdout test split ({len(test_df)} rows) to {test_save_path}")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    # Preprocessor
    preprocessor = build_preprocessor()

    # 1. Baseline Model: Logistic Regression
    baseline_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, class_weight='balanced', random_state=42))
    ])

    # 2. Production Model: Random Forest Classifier
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n" + "=" * 60)
    print("STEP 2: 5-Fold Stratified Cross-Validation")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline Evaluation
    baseline_cv_scores = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"Baseline (Logistic Regression) Mean Macro-F1 : {baseline_cv_scores.mean():.4f} (+/- {baseline_cv_scores.std():.4f})")

    # Production Model Evaluation
    rf_cv_scores = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"Production (Random Forest)     Mean Macro-F1 : {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

    # Train Final Random Forest on all training data
    print("\nFitting final production pipeline on entire training partition...")
    rf_pipe.fit(X_train, y_train)

    # Save Trained Model
    model_path = os.path.join(models_dir, "employee_perf_model.pkl")
    joblib.dump(rf_pipe, model_path)
    print(f"Production model successfully serialized to: {model_path}")

    # Extract One-Hot Encoded Feature Names
    fitted_pre = rf_pipe.named_steps['preprocessor']
    cat_encoder = fitted_pre.named_transformers_['cat'].named_steps['ohe']
    cat_feature_names = list(cat_encoder.get_feature_names_out(get_feature_lists()['categorical']))
    all_feature_names = get_feature_lists()['numerical'] + cat_feature_names

    metadata = {
        'model_type': 'RandomForestClassifier',
        'n_features_in': len(all_feature_names),
        'features': all_feature_names,
        'classes': list(rf_pipe.classes_),
        'baseline_cv_f1_macro': float(baseline_cv_scores.mean()),
        'production_cv_f1_macro': float(rf_cv_scores.mean())
    }

    metadata_path = os.path.join(models_dir, "feature_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Feature metadata saved to: {metadata_path}")
    print("=" * 60)

    return rf_pipe

if __name__ == '__main__':
    run_training()
