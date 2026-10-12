"""
HR Decision & Actionable Recommendation Engine
Author: Data Science & HR Analytics Team
Purpose: Maps predictive bands and employee feature signals into concrete,
         personalized HR interventions, coaching roadmaps, and review timelines.
"""

def generate_employee_recommendation(emp_row, predicted_band, proba_dict=None):
    """
    Evaluates individual employee feature metrics and returns:
    1. Key drivers (strengths & bottlenecks)
    2. Specific HR intervention action
    3. Review milestone horizon
    """
    strengths = []
    risks = []
    actions = []

    # 1. Delivery & Task Management
    otd = emp_row.get('on_time_delivery_rate', 0.8)
    task_delay = emp_row.get('avg_task_delay_days', 0)
    if otd >= 0.90:
        strengths.append(f"High on-time delivery rate ({otd * 100:.0f}%)")
    elif otd < 0.70:
        risks.append(f"Suboptimal delivery consistency ({otd * 100:.0f}%)")
        actions.append("Enroll in Agile Sprint Planning & Workload Estimation Workshop")

    if task_delay > 6.0:
        risks.append(f"Elevated average task delay (+{task_delay:.1f} days)")
        actions.append("Conduct bi-weekly sprint blocker check-ins with Engineering Manager")
    elif task_delay <= 0.0:
        strengths.append("Delivers tasks on or ahead of scheduled deadlines")

    # 2. Learning & Development (Training)
    training = emp_row.get('training_hours', 0)
    certs = emp_row.get('certifications_count', 0)
    if training < 15:
        risks.append(f"Low L&D engagement ({training} training hours)")
        actions.append("Mandate 20 hours of domain-specific technical upskilling")
    elif training >= 40:
        strengths.append(f"Strong commitment to upskilling ({training} training hours, {certs} certs)")

    # 3. Attendance & Work Discipline
    absences = emp_row.get('unplanned_absences', 0)
    if absences >= 6:
        risks.append(f"High unplanned absence rate ({absences} days)")
        actions.append("HR 1-on-1 wellness check-in to assess burnout / work-life balance")

    # 4. Collaboration & Feedback
    peer_score = emp_row.get('peer_feedback_score', 3.0)
    mgr_score = emp_row.get('manager_score', 3.0)
    kudos = emp_row.get('kudos_count', 0)

    if peer_score >= 4.2 and mgr_score >= 4.2:
        strengths.append(f"Exceptional stakeholder feedback (Peer: {peer_score}/5, Mgr: {mgr_score}/5)")
    if kudos >= 10:
        strengths.append(f"High peer recognition ({kudos} kudos received)")

    if mgr_score < 2.5:
        risks.append(f"Manager performance rating concerns ({mgr_score}/5)")
        actions.append("Establish explicit 60-day Performance Improvement Plan (PIP) milestones")

    # Overall Decision Synthesis based on Band
    if predicted_band == 'High':
        primary_decision = "Eligible for Promotion / High-Potential (HiPo) Fast Track"
        if not actions:
            actions.append("Nominate for Leadership Mentorship Program & High-Impact Strategic Projects")
        timeline = "Quarterly Appraisal Calibration"
    elif predicted_band == 'Low':
        primary_decision = "At-Risk Employee: Immediate Targeted Coaching & PIP Support Required"
        if not actions:
            actions.append("Conduct structured performance alignment review with Department Head")
        timeline = "30-Day Checkpoint"
    else: # Medium
        primary_decision = "Steady Performer: Solid Baseline with Targeted Upskilling Opportunities"
        if not actions:
            actions.append("Set stretch goals for the upcoming quarter to bridge gap to Senior band")
        timeline = "Mid-Year Review"

    return {
        'predicted_band': predicted_band,
        'primary_decision': primary_decision,
        'top_strengths': strengths[:3] if strengths else ["Consistent baseline performance across metrics"],
        'key_bottlenecks': risks[:3] if risks else ["No critical operational bottlenecks identified"],
        'recommended_actions': actions[:3],
        'review_timeline': timeline
    }
