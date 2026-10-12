"""
Virtual Enterprise Simulation Script
Author: Data Science & HR Analytics Team
Purpose: Simulates a real-world enterprise People Analytics workflow across 4 quarters:
         Data ingestion -> Automated Risk Scoring -> Manager Calibration -> Intervention.
"""

import time
import os
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.predict import predict_single_employee, predict_batch_csv

def display_banner(text):
    print("\n" + "#" * 75)
    print(f"  {text}")
    print("#" * 75)

def run_corporate_simulation():
    display_banner("ACME CORP - ENTERPRISE HR PERFORMANCE ANALYTICS SIMULATION")
    time.sleep(1)

    print("\n[SIMULATION SCENARIO]")
    print("Company: Acme Global Technologies (2,500 active employees)")
    print("Cycle: Q3 Pre-Appraisal Early Warning & Calibration (60 Days Before Year-End)")
    print("Objective: Predict Q4 performance ratings, identify retention risks & coaching needs.")

    time.sleep(1)
    display_banner("STEP 1: AGGREGATING WORKPLACE TELEMETRY & FEEDBACK DATA")
    print("Connecting to simulated company endpoints:")
    print("  [>] Jira / Azure DevOps : story points, sprint delivery, bug counts")
    print("  [>] HRMS / Workday      : tenure, department, salary percentile, absences")
    print("  [>] LMS (Cornerstone)   : training hours completed, technical certs")
    print("  [>] 360 Feedback Engine : manager scores, peer reviews, kudos")
    print("Status: 2,500 employee behavioral telemetry records ingested successfully.")

    time.sleep(1)
    display_banner("STEP 2: MODEL INFERENCE & STRATIFIED RISK SCORING")
    print("Running production Random Forest classifier pipeline (models/employee_perf_model.pkl)...")
    
    sim_batch_df = predict_batch_csv(
        input_csv="data/test_sample_unseen.csv",
        output_csv="outputs/simulation_q3_intervention_summary.csv"
    )

    time.sleep(1)
    display_banner("STEP 3: HR TALENT CALIBRATION COMMITTEE REVIEW")
    print("Simulating HR Director & Engineering VP review session on key profiles:\n")

    for idx, row in sim_batch_df.iterrows():
        emp_id = row['employee_id']
        dept = row['department']
        band = row['predicted_performance']
        conf = row['confidence']
        decision = row['hr_decision']
        actions = row['recommended_actions']
        timeline = row['review_timeline']

        print(f"[-] EMPLOYEE: {emp_id} | Department: {dept}")
        print(f"    - Predicted Band   : [{band.upper()}] (Confidence: {conf})")
        print(f"    - Strategic Action : {decision}")
        print(f"    - Coaching Steps   : {actions}")
        print(f"    - Next Milestone   : {timeline}")
        print("-" * 75)
        time.sleep(0.5)

    display_banner("STEP 4: BUSINESS VALUE & ROI SUMMARY")
    print("Projected Enterprise Outcomes for Acme Corp:")
    print("  1. Unplanned Attrition Reduction : 18% lower turnover of High Performers (early rewards).")
    print("  2. Performance Turnaround Rate   : 34% of Low Performers successfully remediated via early PIP.")
    print("  3. Calibration Time Saved        : HR appraisal meeting cycle reduced from 3 weeks to 4 days.")
    print("  4. L&D Budget Efficiency         : Training spend redirected accurately to specific skill gaps.")
    print("=" * 75)
    print("Simulation completed. Review file: outputs/simulation_q3_intervention_summary.csv\n")

if __name__ == '__main__':
    run_corporate_simulation()
