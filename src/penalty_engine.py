import numpy as np
from src.config import CONFIG, CONSULTING_FIRMS, LD_KEYWORDS, PRE_LLM_PATTERN, PRE_LLM_CUTOFF_DATE

def apply_penalties(evidence_list: list) -> np.ndarray:
    """
    Phase 6: Penalty Engine
    Uses soft tiered penalties: -0.05, -0.10, -0.20.
    Bounded strictly at MAX_PENALTY (e.g. -0.50) which acts as a hard filter.
    Returns an array of penalties to subtract from the tech score.
    """
    N = len(evidence_list)
    penalties = np.zeros(N, dtype=np.float32)

    for i, ev in enumerate(evidence_list):
        penalty = 0.0
        raw = ev["_source_candidate"]
        title = ev["current_title"].lower()
        
        career_blob = " ".join([r.get('description', '').lower() for r in raw.get('career_history', [])])
        skills_blob = " ".join([s["name"].lower() for s in ev["top_skills"]])
        full_text = career_blob + " " + skills_blob + " " + title
        
        # --- HARD FILTERS (-0.50) ---
        
        # 1. consulting_only
        career = raw.get('career_history', [])
        is_consulting_only = False
        if len(career) > 0:
            all_consulting = all(any(c in r.get('company', '').lower() for c in CONSULTING_FIRMS) for r in career)
            no_prod = len(ev["production_found"]) == 0
            if all_consulting and no_prod:
                is_consulting_only = True
                
        if is_consulting_only:
            penalty += 0.50
            ev["penalties"].append("-0.50 (HARD FILTER): Consulting-only career with no production evidence")
            
        # 2. pure_research (check both production_found AND career text for production signals)
        is_research_title = any(w in title for w in ["research", "scientist", "academic", "professor"])
        prod_keywords_in_text = any(k in career_blob for k in ["deployed", "shipped", "production", "at scale", "latency", "sla"])
        has_prod_evidence = len(ev["production_found"]) > 0 or prod_keywords_in_text
        if is_research_title and not has_prod_evidence:
            penalty += 0.50
            ev["penalties"].append("-0.50 (HARD FILTER): Pure research/academic with no production evidence")
            
        # 3. cv_speech_robotics
        is_cv_speech = any(w in title for w in ["computer vision", "speech", "robotics", "cv engineer"])
        has_retrieval = len(ev["retrieval_found"]) > 0 or any(k in full_text for k in ["search", "ranking", "recsys", "recommendation"])
        if is_cv_speech and not has_retrieval:
            penalty += 0.50
            ev["penalties"].append("-0.50 (HARD FILTER): CV/Speech/Robotics without Retrieval/Ranking experience")
            
        # 4. langchain_only
        pre_llm = False
        for role in career:
            if role.get("start_date", "") < PRE_LLM_CUTOFF_DATE:
                role_desc = role.get("description", "").lower()
                role_title = role.get("title", "").lower()
                if PRE_LLM_PATTERN.search(role_desc) or PRE_LLM_PATTERN.search(role_title):
                    pre_llm = True
                    break
                    
        has_ld = any(k in full_text for k in LD_KEYWORDS)
        if ev["jd_match_count"] == 0 and has_ld and not pre_llm:
            penalty += 0.50
            ev["penalties"].append("-0.50 (HARD FILTER): LangChain/OpenAI wrapper only with no pre-LLM ML experience")

        # --- SOFT PENALTIES ---

        # ChatGPT trap (narrowed)
        # Only penalize if they say exactly "experimented with chatgpt" or "played with chatgpt" and no prod
        if ("experimented with chatgpt" in full_text or "played with chatgpt" in full_text) and len(ev["production_found"]) == 0:
            penalty += 0.10
            ev["penalties"].append("-0.10: ChatGPT experimentation trap without production ML")

        # Timeline mismatch (-0.20)
        if ev["years_exp"] >= 4 and ev["total_career_months"] > 0:
            implied_years = ev["total_career_months"] / 12.0
            if implied_years < ev["years_exp"] * 0.35:
                penalty += 0.20
                ev["penalties"].append(f"-0.20: timeline mismatch ({ev['years_exp']}y claimed vs {implied_years:.1f}y history)")
                
        # Expert proficiency with 0 months (-0.20)
        expert_zero = sum(1 for s in ev["top_skills"] if s["proficiency"] in ("expert", "advanced") and s["duration_months"] == 0)
        if expert_zero >= 3:
            penalty += 0.20
            ev["penalties"].append(f"-0.20: honeypot {expert_zero} expert skills with 0 months")
            
        # Notice period > 120 days (-0.05)
        if ev["notice_days"] > 120:
            penalty += 0.05
            ev["penalties"].append("-0.05: excessively long notice period (>120 days)")

        # Cap penalty explicitly
        capped_penalty = min(penalty, CONFIG["MAX_PENALTY"])
        penalties[i] = capped_penalty
        
    return penalties
