"""
Inference & Prediction Pipeline
Author: Data Science & HR Analytics Team
Purpose: Generates rating predictions, class probabilities, and automated HR action plans
         for single employee profiles or batch CSV datasets.
"""

import os
import joblib
import pandas as pd
from src.hr_recommender import generate_employee_recommendation

def predict_single_employee(emp_dict, model_path="models/employee_perf_model.pkl"):
    """
    Takes an employee feature dictionary and returns predicted band, probabilities,
    and tailored HR recommendations.
    """
    model = joblib.load(model_path)
    df_row = pd.DataFrame([emp_dict])
    
    predicted_band = model.predict(df_row)[0]
    probabilities = model.predict_proba(df_row)[0]
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probabilities)}
    
    recommendation = generate_employee_recommendation(emp_dict, predicted_band, prob_dict)
    
    return {
        'employee_id': emp_dict.get('employee_id', 'UNKNOWN'),
        'predicted_band': predicted_band,
        'confidence': max(prob_dict.values()),
        'probabilities': prob_dict,
        'recommendation': recommendation
    }

def predict_batch_csv(input_csv="data/test_sample_unseen.csv",
                      model_path="models/employee_perf_model.pkl",
                      output_csv="outputs/batch_predictions_with_recommendations.csv"):
    """
    Scores an entire batch CSV of employees, appends probabilities and HR action plans,
    and saves the enriched dataset to disk.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found.")

    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    
    preds = model.predict(df)
    probas = model.predict_proba(df)
    classes = list(model.classes_)
    
    results = []
    for idx, row in df.iterrows():
        emp_dict = row.to_dict()
        pred_band = preds[idx]
        prob_dict = {cls: round(float(probas[idx][i]), 3) for i, cls in enumerate(classes)}
        rec = generate_employee_recommendation(emp_dict, pred_band, prob_dict)
        
        results.append({
            'employee_id': emp_dict.get('employee_id', f'EMP_{idx}'),
            'department': emp_dict.get('department', 'N/A'),
            'job_level': emp_dict.get('job_level', 'N/A'),
            'predicted_performance': pred_band,
            'confidence': f"{max(prob_dict.values()) * 100:.1f}%",
            'prob_High': prob_dict.get('High', 0),
            'prob_Medium': prob_dict.get('Medium', 0),
            'prob_Low': prob_dict.get('Low', 0),
            'hr_decision': rec['primary_decision'],
            'recommended_actions': " | ".join(rec['recommended_actions']),
            'review_timeline': rec['review_timeline']
        })
        
    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print(f"\nBatch predictions successfully generated for {len(result_df)} employees.")
    print(f"Results saved to: {output_csv}\n")
    return result_df

if __name__ == '__main__':
    batch_res = predict_batch_csv()
    print(batch_res[['employee_id', 'department', 'predicted_performance', 'confidence', 'hr_decision']])
