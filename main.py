"""
Main Orchestrator: Employee Performance Predictor
Author: Data Science & HR Analytics Team
Usage:
    python main.py             # Runs the complete end-to-end industry pipeline
    python main.py --demo      # Runs instant inference demonstration
"""

import sys
import os

# Ensure safe UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.train import run_training
from src.evaluate import run_evaluation
from src.predict import predict_batch_csv, predict_single_employee

def run_end_to_end_pipeline():
    print("=" * 70)
    print("[*] EMPLOYEE PERFORMANCE PREDICTOR - END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Verify Dataset
    data_path = "data/employee_features.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Generating synthetic HR dataset...")
        from data.generate_data import generate_employee_dataset, generate_unseen_simulation_samples
        os.makedirs("data", exist_ok=True)
        df = generate_employee_dataset(n_samples=2500)
        df.to_csv(data_path, index=False)
        sim_df = generate_unseen_simulation_samples()
        sim_df.to_csv("data/test_sample_unseen.csv", index=False)
        print("Data generated successfully!")

    # 2. Train Model
    print("\n--- PHASE 1: MODEL TRAINING & CROSS-VALIDATION ---")
    run_training()

    # 3. Evaluate Model & Audit Bias
    print("\n--- PHASE 2: EVALUATION, SHAP/IMPORTANCE & FAIRNESS AUDIT ---")
    eval_results = run_evaluation()

    # 4. Batch Prediction on Unseen Cohort
    print("\n--- PHASE 3: BATCH PREDICTION & HR RECOMMENDATIONS ---")
    batch_results = predict_batch_csv()

    # 5. Interactive Demo Profile Check
    print("\n--- PHASE 4: LIVE INDIVIDUAL INFERENCE DEMO ---")
    sample_employee = {
        'employee_id': 'EMP_DEMO_JOHN_DOE',
        'department': 'Engineering',
        'education': 'Bachelors',
        'job_level': 'Senior',
        'gender': 'Male',
        'age': 31,
        'experience_years': 7.5,
        'manager_tenure': 2.0,
        'salary_percentile_band': 'Tier-3 (Senior)',
        'projects_count': 9,
        'avg_task_delay_days': -2.0,
        'on_time_delivery_rate': 0.95,
        'story_points_completed': 140,
        'billable_hours_ratio': 0.92,
        'training_hours': 45,
        'certifications_count': 3,
        'sick_days': 3,
        'unplanned_absences': 0,
        'avg_login_hours': 8.8,
        'peer_feedback_score': 4.7,
        'manager_score': 4.6,
        'kudos_count': 15,
        'promotions_in_2y': 0
    }
    demo_out = predict_single_employee(sample_employee)
    print(f"Employee ID        : {demo_out['employee_id']}")
    print(f"Predicted Band     : {demo_out['predicted_band']}")
    print(f"Confidence Score   : {demo_out['confidence'] * 100:.1f}%")
    print(f"Probabilities      : {demo_out['probabilities']}")
    print(f"HR Decision        : {demo_out['recommendation']['primary_decision']}")
    print(f"Key Strengths      : {', '.join(demo_out['recommendation']['top_strengths'])}")
    print(f"Recommended Action : {', '.join(demo_out['recommendation']['recommended_actions'])}")
    print(f"Review Timeline    : {demo_out['recommendation']['review_timeline']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("Generated Artifacts:")
    print("  - Model Artifact   : models/employee_perf_model.pkl")
    print("  - Evaluation Charts: outputs/confusion_matrix.png, outputs/feature_importance.png")
    print("  - Metrics Report   : outputs/metrics_report.txt")
    print("  - Prescriptive CSV : outputs/batch_predictions_with_recommendations.csv")
    print("=" * 70)

if __name__ == '__main__':
    run_end_to_end_pipeline()
