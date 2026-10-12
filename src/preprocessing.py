"""
Data Preprocessing & Pipeline Module
Author: Data Science & HR Analytics Team
Purpose: Robust feature separation, categorical encoding, imputation, scaling,
         and ethical feature filtering (excluding sensitive demographic attributes).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET_COL = 'perf_band_next'
ID_COL = 'employee_id'
SENSITIVE_COLS = ['gender', 'age']

# Feature Categorization
CATEGORICAL_FEATURES = [
    'department',
    'education',
    'job_level',
    'salary_percentile_band'
]

NUMERICAL_FEATURES = [
    'experience_years',
    'manager_tenure',
    'projects_count',
    'avg_task_delay_days',
    'on_time_delivery_rate',
    'story_points_completed',
    'billable_hours_ratio',
    'training_hours',
    'certifications_count',
    'sick_days',
    'unplanned_absences',
    'avg_login_hours',
    'peer_feedback_score',
    'manager_score',
    'kudos_count',
    'promotions_in_2y'
]

def build_preprocessor():
    """
    Builds a scikit-learn ColumnTransformer that imputes and scales numerical
    features using RobustScaler, and imputes and one-hot encodes categoricals.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Automatically drops employee_id, gender, age from training
    )
    return preprocessor

def get_feature_lists():
    return {
        'numerical': NUMERICAL_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'sensitive': SENSITIVE_COLS,
        'target': TARGET_COL,
        'id': ID_COL
    }
