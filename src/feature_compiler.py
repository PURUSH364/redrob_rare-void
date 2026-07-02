import numpy as np
from datetime import datetime
from src.config import (NUM_FEATURES, PREFERRED_CITIES_TIER1, PREFERRED_CITIES_TIER2,
                        CONSULTING_FIRMS, PRE_LLM_PATTERN, GOOD_TITLES, BAD_TITLES,
                        RELEVANT_INDUSTRIES, REFERENCE_DATE, PRE_LLM_CUTOFF_DATE,
                        YOE_FIT_CENTER, YOE_FIT_SIGMA, RECENCY_MAX_DAYS,
                        RECENCY_DECAY_FACTOR, RECENCY_FLOOR, SALARY_RANGE_MIN,
                        SALARY_RANGE_MAX)
from src.aliases import COMPILED_ALIASES

# Feature indices
F_VECTOR_DB = 0
F_RETRIEVAL_RAG = 1
F_EVAL = 2
F_PYTHON = 3
F_RECSYS = 4
F_LTR = 5
F_FINE_TUNING = 6
F_PROD_DEPTH = 7
F_PRODUCT_CO = 8
F_YOE_FIT = 9
F_TITLE_ALIGN = 10
F_RECENCY = 11
F_TENURE = 12
F_SHIPPED_SYS = 13
F_INDUSTRY_FIT = 14
F_LOCATION_FIT = 15
F_EDUCATION_FIT = 16
F_WORK_MODE_FIT = 17
F_SALARY_FIT = 18
F_PRE_LLM_ML = 19

def _weighted_skill_score(found_list, skill_duration_map, career_blob):
    """Score skills by depth: duration-weighted with career-context bonus.
    Replaces pure keyword counting to reward demonstrated experience."""
    score = 0.0
    for item in found_list:
        max_dur = 0
        for skill_name, dur in skill_duration_map.items():
            if item in skill_name or skill_name in item:
                max_dur = max(max_dur, dur)
        in_career = item in career_blob
        if max_dur > 0 and in_career:
            dur_factor = min(max_dur / 36.0, 1.0)
            score += 0.5 + 0.5 * dur_factor
        elif in_career:
            score += 0.35
        elif max_dur > 0:
            dur_factor = min(max_dur / 36.0, 1.0)
            score += 0.25 * dur_factor
    return score


def _alias_skill_depth(alias_key, skill_duration_map, career_blob):
    """Score a skill group using compiled alias regex for broad matching."""
    if alias_key not in COMPILED_ALIASES:
        return 0.0
    pattern = COMPILED_ALIASES[alias_key]
    in_career = bool(pattern.search(career_blob))
    max_dur = 0
    for skill_name, dur in skill_duration_map.items():
        if pattern.search(skill_name):
            max_dur = max(max_dur, dur)
    if max_dur > 0 and in_career:
        return min(0.5 + 0.5 * min(max_dur / 36.0, 1.0), 1.0)
    elif in_career:
        return 0.35
    elif max_dur > 0:
        return min(0.25 * min(max_dur / 36.0, 1.0), 1.0)
    return 0.0


def compile_feature_matrix(evidence_list: list) -> np.ndarray:
    N = len(evidence_list)
    matrix = np.zeros((N, NUM_FEATURES), dtype=np.float32)

    for i, ev in enumerate(evidence_list):
        row = matrix[i]
        raw = ev["_source_candidate"]
        
        yoe = ev["years_exp"]
        
        # --- Keyword Stuffing Defense ---
        valid_skills = []
        expert_count = 0
        for s in ev["top_skills"]:
            dur = s["duration_months"]
            prof = s["proficiency"]
            
            if prof == "expert":
                expert_count += 1
                
            # Discard obvious stuffers
            if dur == 0:
                continue
            if prof == "beginner" and dur < 6:
                continue
                
            valid_skills.append(s["name"])
            
        skill_cap = 0.5 if (expert_count >= 5 and yoe < 3) else 1.0

        # Career text blob for depth matching (defined early for all feature use)
        career_blob = " ".join([r.get('description', '').lower() for r in raw.get('career_history', [])])

        # Build duration lookup from validated skills
        skill_duration_map = {}
        for s in ev["top_skills"]:
            name = s["name"].lower()
            dur = s["duration_months"]
            if dur > 0:
                skill_duration_map[name] = max(skill_duration_map.get(name, 0), dur)

        # F0-F3: Critical (depth-weighted scoring)
        row[F_VECTOR_DB] = min(_weighted_skill_score(ev["vector_db_found"], skill_duration_map, career_blob) / 2.0, 1.0) * skill_cap
        row[F_RETRIEVAL_RAG] = min(_weighted_skill_score(ev["retrieval_found"], skill_duration_map, career_blob) / 1.5, 1.0) * skill_cap
        row[F_EVAL] = min(_weighted_skill_score(ev["eval_found"], skill_duration_map, career_blob) / 1.5, 1.0) * skill_cap
        row[F_PYTHON] = min(_weighted_skill_score(ev["python_found"], skill_duration_map, career_blob) / 1.5, 1.0) * skill_cap

        # F4-F6: Strong skills (alias-expanded, depth-weighted)
        row[F_RECSYS] = min(_alias_skill_depth("recsys", skill_duration_map, career_blob), 1.0) * skill_cap
        row[F_LTR] = min(_alias_skill_depth("ltr", skill_duration_map, career_blob), 1.0) * skill_cap
        row[F_FINE_TUNING] = min(_alias_skill_depth("fine_tuning", skill_duration_map, career_blob), 1.0) * skill_cap
        
        # F7: prod_depth
        row[F_PROD_DEPTH] = min(len(ev["production_found"]) / 5.0, 1.0)
        
        # F8: product_co_ratio (actual ratio of non-consulting tenure)
        total_months = max(ev["total_career_months"], 1)
        consulting_months = 0
        for role in raw.get('career_history', []):
            company = role.get('company', '').lower()
            if any(c in company for c in CONSULTING_FIRMS):
                consulting_months += int(role.get('duration_months', 0) or 0)
        product_ratio = 1.0 - (consulting_months / total_months)
        row[F_PRODUCT_CO] = max(min(product_ratio, 1.0), 0.0)
        
        # F9: yoe_fit
        row[F_YOE_FIT] = min(np.exp(-0.5 * ((yoe - YOE_FIT_CENTER) / YOE_FIT_SIGMA) ** 2), 1.0)
        
        # F10: title_align
        title = ev["current_title"].lower()
        if any(b in title for b in BAD_TITLES):
            row[F_TITLE_ALIGN] = 0.0
        elif any(g in title for g in GOOD_TITLES):
            row[F_TITLE_ALIGN] = 1.0
        else:
            row[F_TITLE_ALIGN] = 0.5
            
        # F11: recency (based on platform activity, not just having a career)
        last_active = ev.get("last_active_date", "")
        if last_active:
            try:
                la_date = datetime.strptime(last_active, "%Y-%m-%d")
                days_inactive = max((REFERENCE_DATE - la_date).days, 0)
                row[F_RECENCY] = max(1.0 - (days_inactive / RECENCY_MAX_DAYS) * RECENCY_DECAY_FACTOR, RECENCY_FLOOR)
            except (ValueError, TypeError):
                row[F_RECENCY] = 0.5  # Unknown activity date
        else:
            row[F_RECENCY] = 0.5
        
        # F12: tenure 
        n_roles = len(raw.get('career_history', []))
        avg_tenure = ev["total_career_months"] / max(n_roles, 1)
        row[F_TENURE] = min(avg_tenure / 24.0, 1.0)
        
        # F13: shipped_systems
        row[F_SHIPPED_SYS] = min(len(ev["production_found"]) / 3.0, 1.0)
        
        # F14: industry_fit (independent check for relevant industry experience)
        industry_hits = sum(1 for ind in RELEVANT_INDUSTRIES if ind in career_blob)
        row[F_INDUSTRY_FIT] = min(industry_hits / 2.0, 1.0)
        
        # F15: location_fit
        loc = ev["location"]
        loc_score = 0.2
        if any(c in loc for c in PREFERRED_CITIES_TIER1):
            loc_score = 1.0
        elif any(c in loc for c in PREFERRED_CITIES_TIER2):
            loc_score = 0.7
        elif "india" in ev["country"]:
            loc_score = 0.4
            
        if ev["willing_to_relocate"] and loc_score < 1.0:
            loc_score = min(loc_score + 0.3, 1.0)
        row[F_LOCATION_FIT] = loc_score

        # F16: education_fit
        edu_score = 0.2
        for edu in ev["education"]:
            field = edu["field_of_study"]
            if any(k in field for k in ["computer science", "artificial intelligence", "machine learning", "nlp", "information retrieval"]):
                curr = 1.0
            elif any(k in field for k in ["engineering", "mathematics", "physics", "data"]):
                curr = 0.6
            else:
                curr = 0.2
                
            if edu["tier"] == "tier_1":
                curr = min(curr + 0.2, 1.0)
            edu_score = max(edu_score, curr)
        row[F_EDUCATION_FIT] = edu_score

        # F17: work_mode_fit
        mode = ev["preferred_work_mode"]
        if mode in ["hybrid", "flexible"]:
            row[F_WORK_MODE_FIT] = 1.0
        elif mode == "onsite":
            row[F_WORK_MODE_FIT] = 0.8
        elif mode == "remote":
            row[F_WORK_MODE_FIT] = 0.6
        else:
            row[F_WORK_MODE_FIT] = 0.5

        # F18: salary_fit
        sal_min, sal_max = ev["expected_salary_min"], ev["expected_salary_max"]
        midpoint = (sal_min + sal_max) / 2.0
        # Reasonable ML engineer range in India ~20-40 LPA depending on exp
        if SALARY_RANGE_MIN <= midpoint <= SALARY_RANGE_MAX:
            row[F_SALARY_FIT] = 1.0
        elif midpoint == 0:
            row[F_SALARY_FIT] = 0.5
        else:
            row[F_SALARY_FIT] = 0.3

        # F19: pre_llm_ml_experience
        pre_llm_bonus = 0.0
        for role in raw.get("career_history", []):
            start = role.get("start_date", "")
            if start and start < PRE_LLM_CUTOFF_DATE:
                desc = role.get("description", "").lower()
                role_title = role.get("title", "").lower()
                if PRE_LLM_PATTERN.search(desc) or PRE_LLM_PATTERN.search(role_title):
                    pre_llm_bonus = 1.0
                    break
        row[F_PRE_LLM_ML] = pre_llm_bonus

    return matrix
