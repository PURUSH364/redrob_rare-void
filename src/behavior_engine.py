import numpy as np
from datetime import datetime
from src.config import CONFIG, BEHAVIOR_WEIGHTS, REFERENCE_DATE

def compute_behavior_multiplier(evidence_list: list) -> np.ndarray:
    """
    Phase 9: Behavior Multiplier
    Extracts behavior signals and computes a multiplier clamped to [BEHAVIOR_MIN, BEHAVIOR_MAX].
    Does NOT dominate the technical score.
    Uses REFERENCE_DATE instead of datetime.now() for deterministic reproducibility.
    """
    N = len(evidence_list)
    multipliers = np.zeros(N, dtype=np.float32)
    
    W_RESPONSE = BEHAVIOR_WEIGHTS["W_RESPONSE"]
    W_GITHUB = BEHAVIOR_WEIGHTS["W_GITHUB"]
    W_AVAILABILITY = BEHAVIOR_WEIGHTS["W_AVAILABILITY"]
    W_DEMAND = BEHAVIOR_WEIGHTS["W_DEMAND"]

    for i, ev in enumerate(evidence_list):
        # 1. Response Rate
        response = min(max(ev["response_rate"], 0.0), 1.0)
        
        # 2. Github (-1 means no github, we treat as 0.2 neutral)
        github = ev["github_score"]
        if github < 0:
            github_norm = 0.20
        else:
            github_norm = min(github / 100.0, 1.0)
            
        # 3. Availability
        otw = 1.0 if ev["open_to_work"] else 0.20
        
        # Inactivity check
        last_active = ev.get("last_active_date", "")
        if last_active:
            try:
                la_date = datetime.strptime(last_active, "%Y-%m-%d")
                days_inactive = (REFERENCE_DATE - la_date).days
                if days_inactive > 180:
                    otw = 0.0 # Highly inactive
            except (ValueError, TypeError):
                pass

        notice = ev["notice_days"]
        notice_factor = max(0.0, 1.0 - min(notice, 90) / 90.0)
        availability = 0.6 * otw + 0.4 * notice_factor
        
        # 4. Demand (saved by recruiters)
        saved = ev["saved_30d"]
        demand_base = min(saved / 10.0, 1.0)
        
        # Base composite [0, 1]
        composite = (
            W_RESPONSE * response +
            W_GITHUB * github_norm +
            W_AVAILABILITY * availability +
            W_DEMAND * demand_base
        )
        
        # Add explicit modifiers to composite
        # avg_response_time_hours: < 24h = +0.15, 24-72h = +0.05, > 168h = -0.10
        rt = ev["avg_response_time_hours"]
        if rt < 24:
            composite += 0.15
        elif 24 <= rt <= 72:
            composite += 0.05
        elif rt > 168:
            composite -= 0.10
            
        # saved_by_recruiters_30d
        if saved > 5:
            composite += 0.10
        elif 1 <= saved <= 5:
            composite += 0.03
        else:
            composite -= 0.05
            
        # profile_completeness_score
        pcs = ev["profile_completeness_score"]
        if pcs > 80:
            composite += 0.05
        elif pcs < 40:
            composite -= 0.05
            
        # interview_completion_rate
        icr = ev["interview_completion_rate"]
        if icr > 0.8:
            composite += 0.05
        elif icr < 0.3:
            composite -= 0.05
            
        # applications_30d
        apps = ev["applications_30d"]
        if apps >= 3:
            composite += 0.05
        elif apps == 0:
            composite -= 0.03

        # Map to multiplier
        # Clip composite back to [0, 1]
        composite_clipped = max(0.0, min(composite, 1.0))
        
        b_min = CONFIG["BEHAVIOR_MIN"]
        b_max = CONFIG["BEHAVIOR_MAX"]
        
        multipliers[i] = b_min + (b_max - b_min) * composite_clipped
        
    return multipliers
