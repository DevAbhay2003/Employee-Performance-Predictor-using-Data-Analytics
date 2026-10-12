"""
Synthetic HR Dataset Generator for Employee Performance Prediction
Author: Data Science & HR Analytics Team
Purpose: Generates realistic employee performance data with realistic behavioral signals,
         avoiding synthetic artifact leakage while enforcing industry HR distribution rules.
"""

import os
import numpy as np
import pandas as pd

def generate_employee_dataset(n_samples=2500, random_seed=42):
    np.random.seed(random_seed)

    departments = ['Engineering', 'Sales', 'Product', 'Marketing', 'Customer Support', 'Finance']
    dept_probs = [0.35, 0.20, 0.12, 0.11, 0.12, 0.10]
    
    education_levels = ['Bachelors', 'Masters', 'PhD', 'Diploma']
    edu_probs = [0.60, 0.28, 0.05, 0.07]
    
    job_levels = ['Junior', 'Mid-Level', 'Senior', 'Lead']
    job_probs = [0.35, 0.35, 0.20, 0.10]
    
    genders = ['Male', 'Female', 'Non-Binary']
    gender_probs = [0.52, 0.45, 0.03]
    
    salary_bands = ['Tier-1 (Entry)', 'Tier-2 (Mid)', 'Tier-3 (Senior)', 'Tier-4 (Lead/Exec)']

    # 1. Base Demographics and Organizational Data
    emp_ids = [f"EMP_{1000 + i}" for i in range(n_samples)]
    dept = np.random.choice(departments, size=n_samples, p=dept_probs)
    edu = np.random.choice(education_levels, size=n_samples, p=edu_probs)
    j_level = np.random.choice(job_levels, size=n_samples, p=job_probs)
    gender = np.random.choice(genders, size=n_samples, p=gender_probs)
    
    # Experience depends on Job Level
    exp_years = []
    ages = []
    salary_band_list = []
    for level in j_level:
        if level == 'Junior':
            exp = np.random.uniform(0.5, 3.5)
            age = int(21 + exp + np.random.uniform(0, 3))
            s_band = np.random.choice(salary_bands, p=[0.75, 0.20, 0.05, 0.00])
        elif level == 'Mid-Level':
            exp = np.random.uniform(3.5, 8.0)
            age = int(24 + exp + np.random.uniform(0, 4))
            s_band = np.random.choice(salary_bands, p=[0.15, 0.70, 0.15, 0.00])
        elif level == 'Senior':
            exp = np.random.uniform(7.5, 15.0)
            age = int(26 + exp + np.random.uniform(0, 5))
            s_band = np.random.choice(salary_bands, p=[0.02, 0.18, 0.70, 0.10])
        else: # Lead
            exp = np.random.uniform(12.0, 24.0)
            age = int(28 + exp + np.random.uniform(0, 6))
            s_band = np.random.choice(salary_bands, p=[0.00, 0.05, 0.35, 0.60])
            
        exp_years.append(round(exp, 1))
        ages.append(min(age, 62))
        salary_band_list.append(s_band)

    manager_tenure = np.round(np.random.uniform(0.5, 6.0, size=n_samples), 1)
    
    # 2. Performance & Behavior Indicators (Last 6-12 Months)
    # Latent employee baseline competence factor (normal dist)
    competence = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # Projects completed
    projects_count = np.clip(np.random.poisson(lam=np.maximum(1.0, 5.0 + competence * 1.5)), 1, 14)
    
    # On-time delivery rate (0.40 to 1.0)
    base_otd = 0.80 + (competence * 0.08) + np.random.normal(0, 0.05, size=n_samples)
    on_time_delivery_rate = np.clip(np.round(base_otd, 2), 0.35, 1.0)
    
    # Average task delay days (-4 days early to +25 days delay)
    delay = 4.0 - (competence * 4.5) + np.random.normal(0, 2.5, size=n_samples)
    avg_task_delay_days = np.round(np.clip(delay, -5.0, 35.0), 1)
    
    # Story points completed / workload delivered
    points = 80 + (competence * 25) + np.random.normal(0, 15, size=n_samples)
    story_points_completed = np.clip(np.round(points, 0), 20, 200).astype(int)
    
    # Billable hours / productivity ratio
    billable = 0.82 + (competence * 0.06) + np.random.normal(0, 0.06, size=n_samples)
    billable_hours_ratio = np.clip(np.round(billable, 2), 0.45, 1.0)
    
    # Learning and Engagement
    train_hrs = 28 + (competence * 12) + np.random.normal(0, 10, size=n_samples)
    training_hours = np.clip(np.round(train_hrs, 0), 2, 90).astype(int)
    
    certs = np.clip(np.random.poisson(lam=np.maximum(0.2, 1.5 + competence * 0.8)), 0, 6)
    
    # Attendance & Work Discipline
    sick = np.clip(np.random.poisson(lam=4.0, size=n_samples), 0, 22)
    unplanned_abs = np.clip(np.random.poisson(lam=np.maximum(0.5, 3.0 - competence * 1.2)), 0, 18)
    avg_login = np.clip(np.round(8.2 + (competence * 0.4) + np.random.normal(0, 0.4, size=n_samples), 1), 6.5, 10.8)
    
    # Human Feedback Signals
    mgr_score = np.clip(np.round(3.3 + (competence * 0.75) + np.random.normal(0, 0.35, size=n_samples), 1), 1.0, 5.0)
    peer_score = np.clip(np.round(3.4 + (competence * 0.65) + np.random.normal(0, 0.40, size=n_samples), 1), 1.0, 5.0)
    kudos = np.clip(np.random.poisson(lam=np.maximum(1.0, 6.0 + competence * 3.5)), 0, 28)
    
    # Historical HR signals
    promotions_in_2y = np.where(competence + np.random.normal(0, 0.5, size=n_samples) > 1.2, 1, 0)

    # 3. Target Label Calculation (Next Performance Cycle Band: High, Medium, Low)
    # Standard multi-factor evaluation composite index:
    composite_score = (
        (on_time_delivery_rate * 25.0) +
        (mgr_score * 7.5) +
        (peer_score * 5.0) +
        (story_points_completed * 0.12) +
        (training_hours * 0.10) -
        (avg_task_delay_days * 0.8) -
        (unplanned_abs * 1.2) +
        (kudos * 0.5) +
        np.random.normal(0, 3.5, size=n_samples) # Real-world subjectivity & noise
    )
    
    q_low = np.percentile(composite_score, 25)
    q_high = np.percentile(composite_score, 75)
    
    perf_band = []
    for score in composite_score:
        if score >= q_high:
            perf_band.append('High')
        elif score <= q_low:
            perf_band.append('Low')
        else:
            perf_band.append('Medium')
            
    df = pd.DataFrame({
        'employee_id': emp_ids,
        'department': dept,
        'education': edu,
        'job_level': j_level,
        'gender': gender,
        'age': ages,
        'experience_years': exp_years,
        'manager_tenure': manager_tenure,
        'salary_percentile_band': salary_band_list,
        'projects_count': projects_count,
        'avg_task_delay_days': avg_task_delay_days,
        'on_time_delivery_rate': on_time_delivery_rate,
        'story_points_completed': story_points_completed,
        'billable_hours_ratio': billable_hours_ratio,
        'training_hours': training_hours,
        'certifications_count': certs,
        'sick_days': sick,
        'unplanned_absences': unplanned_abs,
        'avg_login_hours': avg_login,
        'peer_feedback_score': peer_score,
        'manager_score': mgr_score,
        'kudos_count': kudos,
        'promotions_in_2y': promotions_in_2y,
        'perf_band_next': perf_band
    })
    
    return df

def generate_unseen_simulation_samples():
    """Generates a small test set of distinct employee profiles for simulation and live testing."""
    test_profiles = [
        {
            'employee_id': 'SIM_001_HIGH',
            'department': 'Engineering',
            'education': 'Masters',
            'job_level': 'Senior',
            'gender': 'Female',
            'age': 32,
            'experience_years': 8.5,
            'manager_tenure': 2.5,
            'salary_percentile_band': 'Tier-3 (Senior)',
            'projects_count': 8,
            'avg_task_delay_days': -1.5,
            'on_time_delivery_rate': 0.96,
            'story_points_completed': 145,
            'billable_hours_ratio': 0.94,
            'training_hours': 48,
            'certifications_count': 3,
            'sick_days': 2,
            'unplanned_absences': 0,
            'avg_login_hours': 8.9,
            'peer_feedback_score': 4.8,
            'manager_score': 4.7,
            'kudos_count': 16,
            'promotions_in_2y': 0
        },
        {
            'employee_id': 'SIM_002_LOW',
            'department': 'Sales',
            'education': 'Bachelors',
            'job_level': 'Junior',
            'gender': 'Male',
            'age': 24,
            'experience_years': 1.2,
            'manager_tenure': 0.8,
            'salary_percentile_band': 'Tier-1 (Entry)',
            'projects_count': 2,
            'avg_task_delay_days': 16.5,
            'on_time_delivery_rate': 0.52,
            'story_points_completed': 38,
            'billable_hours_ratio': 0.60,
            'training_hours': 6,
            'certifications_count': 0,
            'sick_days': 12,
            'unplanned_absences': 9,
            'avg_login_hours': 7.1,
            'peer_feedback_score': 2.1,
            'manager_score': 2.2,
            'kudos_count': 1,
            'promotions_in_2y': 0
        },
        {
            'employee_id': 'SIM_003_MED',
            'department': 'Product',
            'education': 'Bachelors',
            'job_level': 'Mid-Level',
            'gender': 'Male',
            'age': 29,
            'experience_years': 5.0,
            'manager_tenure': 2.0,
            'salary_percentile_band': 'Tier-2 (Mid)',
            'projects_count': 5,
            'avg_task_delay_days': 3.2,
            'on_time_delivery_rate': 0.81,
            'story_points_completed': 82,
            'billable_hours_ratio': 0.80,
            'training_hours': 26,
            'certifications_count': 1,
            'sick_days': 5,
            'unplanned_absences': 2,
            'avg_login_hours': 8.2,
            'peer_feedback_score': 3.6,
            'manager_score': 3.5,
            'kudos_count': 5,
            'promotions_in_2y': 0
        },
        {
            'employee_id': 'SIM_004_TALENT_AT_RISK',
            'department': 'Marketing',
            'education': 'Masters',
            'job_level': 'Mid-Level',
            'gender': 'Female',
            'age': 28,
            'experience_years': 4.5,
            'manager_tenure': 1.2,
            'salary_percentile_band': 'Tier-2 (Mid)',
            'projects_count': 6,
            'avg_task_delay_days': 1.0,
            'on_time_delivery_rate': 0.90,
            'story_points_completed': 95,
            'billable_hours_ratio': 0.88,
            'training_hours': 8, # Needs upskilling
            'certifications_count': 0,
            'sick_days': 3,
            'unplanned_absences': 1,
            'avg_login_hours': 8.4,
            'peer_feedback_score': 4.1,
            'manager_score': 3.9,
            'kudos_count': 9,
            'promotions_in_2y': 0
        }
    ]
    return pd.DataFrame(test_profiles)

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Main Training Dataset
    main_df = generate_employee_dataset(n_samples=2500, random_seed=42)
    output_path = os.path.join(data_dir, 'employee_features.csv')
    main_df.to_csv(output_path, index=False)
    print(f"Generated {len(main_df)} records saved to {output_path}")
    print("Class Distribution:")
    print(main_df['perf_band_next'].value_counts(normalize=True))
    
    # 2. Simulation Test Profiles
    sim_df = generate_unseen_simulation_samples()
    sim_path = os.path.join(data_dir, 'test_sample_unseen.csv')
    sim_df.to_csv(sim_path, index=False)
    print(f"Generated {len(sim_df)} test simulation records saved to {sim_path}")
