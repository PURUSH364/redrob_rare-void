import re
from src.jd_matrix import JD_MATRIX
from src.aliases import COMPILED_ALIASES
from src.config import CONSULTING_FIRMS

def extract_evidence(candidate: dict) -> dict:
    """
    Phase 4: Evidence Builder
    ONLY extract. Never score.
    Returns lists of found strings and provenance, not booleans.
    """
    profile = candidate.get('profile', {}) or {}
    skills = candidate.get('skills', []) or []
    career = candidate.get('career_history', []) or []
    signals = candidate.get('redrob_signals', {}) or {}
    education = candidate.get('education', []) or []

    # Extract raw texts
    skill_names = [s.get('name', '').lower() for s in skills if s.get('name')]
    career_texts = [r.get('description', '') or '' for r in career]
    career_text_blob = " ".join(career_texts).lower()

    # Provenance tracking dictionaries
    vector_db_found = []
    vector_db_sources = {}
    retrieval_found = []
    eval_found = []
    python_found = []
    production_found = []
    
    # 1. Check Vector DBs (Critical)
    for vdb in JD_MATRIX["critical"]["vector_db"]:
        if any(vdb in sn for sn in skill_names):
            vector_db_found.append(vdb)
            vector_db_sources[vdb] = "skills"
        elif vdb in career_text_blob:
            vector_db_found.append(vdb)
            vector_db_sources[vdb] = "career_history"

    # 2. Check Retrieval/RAG (Critical)
    for r_alias in JD_MATRIX["critical"]["retrieval"]:
        matched = False
        if r_alias in COMPILED_ALIASES:
            if any(COMPILED_ALIASES[r_alias].search(sn) for sn in skill_names):
                matched = True
            elif COMPILED_ALIASES[r_alias].search(career_text_blob):
                matched = True
        else:
            if any(r_alias in sn for sn in skill_names) or r_alias in career_text_blob:
                matched = True
        if matched:
            retrieval_found.append(r_alias)

    # 3. Check Evaluation (Critical)
    for e_alias in JD_MATRIX["critical"]["evaluation"]:
        matched = False
        if e_alias in COMPILED_ALIASES:
            if any(COMPILED_ALIASES[e_alias].search(sn) for sn in skill_names) or COMPILED_ALIASES[e_alias].search(career_text_blob):
                matched = True
        else:
            if any(e_alias in sn for sn in skill_names) or e_alias in career_text_blob:
                matched = True
        if matched:
            eval_found.append(e_alias)

    # 4. Python
    for py in JD_MATRIX["critical"]["python"]:
        if any(py in sn for sn in skill_names) or py in career_text_blob:
            python_found.append(py)

    # 5. Production Evidence
    if "production_evidence" in COMPILED_ALIASES:
        matches = COMPILED_ALIASES["production_evidence"].findall(career_text_blob)
        production_found = list(set([m.lower() for m in matches if m]))

    # JD Match Count
    jd_match_count = len(vector_db_found) + len(retrieval_found) + len(eval_found) + len(python_found)
    
    # Career snippets for reasoning
    career_snippets = []
    total_career_months = 0
    for role in career[:4]:
        dur = int(role.get('duration_months', 0) or 0)
        total_career_months += dur
        desc = role.get('description', '') or ''
        # Truncate nicely at word boundaries
        snippet = desc[:100]
        if len(desc) > 100:
            last_space = snippet.rfind(' ')
            if last_space > 0:
                snippet = snippet[:last_space]
            snippet += "..."
            
        career_snippets.append({
            'company': role.get('company', ''),
            'title': role.get('title', ''),
            'start_date': role.get('start_date', ''),
            'end_date': role.get('end_date', ''),
            'duration_months': dur,
            'is_current': bool(role.get('is_current', False)),
            'snippet': snippet,
        })
        
    yoe = float(profile.get('years_of_experience', 0) or 0)
    
    # Skills logic
    top_skills = [
        {
            'name': s.get('name', ''),
            'proficiency': s.get('proficiency', 'beginner'),
            'duration_months': int(s.get('duration_months', 0) or 0),
            'endorsements': int(s.get('endorsements', 0) or 0)
        }
        for s in skills
    ]

    # Education parsing
    edu_list = []
    for edu in education:
        edu_list.append({
            'field_of_study': edu.get('field_of_study', '').lower(),
            'tier': edu.get('tier', 'unknown').lower(),
            'degree': edu.get('degree', '').lower()
        })

    # Salary parsing
    salary = signals.get('expected_salary_range_inr_lpa', {}) or {}
    sal_min = float(salary.get('min', 0) or 0)
    sal_max = float(salary.get('max', 0) or 0)

    evidence = {
        "id": candidate.get("candidate_id"),
        "headline": profile.get("headline", ""),
        "years_exp": yoe,
        "current_title": profile.get("current_title", ""),
        "total_career_months": total_career_months,
        "location": profile.get("location", "").lower(),
        "country": profile.get("country", "").lower(),
        
        "vector_db_found": vector_db_found,
        "vector_db_sources": vector_db_sources,
        "retrieval_found": retrieval_found,
        "eval_found": eval_found,
        "python_found": python_found,
        "production_found": production_found,
        "jd_match_count": jd_match_count,
        
        "career_snippets": career_snippets,
        "top_skills": top_skills,
        "education": edu_list,
        
        # Behavior / Logistics raw signals
        "response_rate": float(signals.get("recruiter_response_rate", 0) or 0),
        "github_score": float(signals.get("github_activity_score", -1) or -1),
        "notice_days": int(signals.get("notice_period_days") or 90),
        "open_to_work": bool(signals.get("open_to_work_flag", False)),
        "saved_30d": int(signals.get("saved_by_recruiters_30d", 0) or 0),
        "preferred_work_mode": signals.get("preferred_work_mode", "").lower(),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "expected_salary_min": sal_min,
        "expected_salary_max": sal_max,
        "avg_response_time_hours": float(signals.get("avg_response_time_hours", 48) or 48),
        "skill_assessment_scores": signals.get("skill_assessment_scores", {}) or {},
        "connection_count": int(signals.get("connection_count", 0) or 0),
        "endorsements_received": int(signals.get("endorsements_received", 0) or 0),
        "search_appearance_30d": int(signals.get("search_appearance_30d", 0) or 0),
        "offer_acceptance_rate": float(signals.get("offer_acceptance_rate", -1) or -1),
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),
        "profile_completeness_score": float(signals.get("profile_completeness_score", 0) or 0),
        "interview_completion_rate": float(signals.get("interview_completion_rate", 0) or 0),
        "applications_30d": int(signals.get("applications_submitted_30d", 0) or 0),
        "last_active_date": signals.get("last_active_date", ""),
        
        # To be populated by Penalty Engine
        "penalties": []
    }
    
    # Explicit pass-through: keep the full candidate dict for downstream modules
    # (feature_compiler, penalty_engine) that need raw career_history and skills data.
    evidence["_source_candidate"] = candidate

    return evidence
