import hashlib

def generate_reasoning(evidence: dict, rank: int = 100) -> str:
    """
    Phase 10: Reason Generator
    Generates deterministic, explainable text from evidence.
    Structurally varied templates — not just word swaps.
    """
    yoe = evidence["years_exp"]
    title = evidence["current_title"] or "Engineer"
    
    company = ""
    if evidence["career_snippets"] and evidence["career_snippets"][0]["company"]:
        company = evidence["career_snippets"][0]["company"]
        
    # Build skills string — pick 1-3 most relevant
    skills = []
    if evidence["vector_db_found"]:
        skills.append(evidence['vector_db_found'][0].title())
    if evidence["retrieval_found"]:
        skills.append("RAG")
    if evidence["eval_found"]:
        skills.append(evidence['eval_found'][0].title())
    skills_str = ", ".join(skills[:3]) if skills else "core ML frameworks"
    
    prod_str = "production ML systems" if evidence["production_found"] else "ML systems"
    
    # Extra facts — sometimes included, sometimes not
    loc = evidence.get("location", "").lower()
    loc_str = ""
    if any(city in loc for city in ["pune", "noida", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad"]):
        loc_str = loc.title()
        
    edu_field = ""
    if evidence.get("education"):
        field = evidence["education"][0]["field_of_study"]
        if any(k in field.lower() for k in ["computer science", "artificial intelligence", "machine learning"]):
            edu_field = field.title()
            
    response = int(evidence["response_rate"] * 100)
    notice = evidence["notice_days"]
    
    concerns = []
    for p in evidence.get("penalties", []):
        parts = p.split(": ")
        concerns.append(parts[1] if len(parts) > 1 else p)
    concern_str = concerns[0] if concerns else ""
    
    # Deterministic selection based on candidate_id
    cid = evidence["id"]
    h = int(hashlib.md5(cid.encode('utf-8')).hexdigest(), 16)
    
    # Collect structural fragments — mix and match differently per template
    def engage_str():
        if response > 80:
            return f"with strong recruiter engagement ({response}%)"
        elif response > 60:
            return f"showing solid recruiter interest ({response}%)"
        return ""
    
    def avail_str():
        if notice <= 30:
            return "available immediately"
        elif notice <= 60:
            return f"available within {notice} days"
        elif notice <= 90:
            return f"with a {notice}-day notice period"
        return ""
    
    def comp_str():
        return f" at {company}" if company else ""
    
    def loc_detail():
        return f"based in {loc_str}" if loc_str else ""
    
    def edu_detail():
        return f"background in {edu_field}" if edu_field else ""
    
    # ---- STRUCTURALLY DISTINCT TEMPLATES ----
    # Each template has a different sentence structure, not just word swaps
    
    if rank <= 10:
        templates = [
            # T1: Lead with skills, then credentials
            f"{skills_str} specialist{comp_str()}. {yoe:.1f} years building {prod_str} {engage_str()}. {f'Education: {edu_detail()}.' if edu_detail() else ''} {'Located in ' + loc_str + '.' if loc_str else ''}".strip(),
            
            # T2: Lead with experience depth, then company
            f"{yoe:.1f}-year veteran{' ' + comp_str() if comp_str() else ''} focused on {prod_str}. Core stack: {skills_str}. {avail_str()}. {f'{edu_detail()}.' if edu_detail() else ''}".strip(),
            
            # T3: Lead with recommendation, then evidence
            f"Top-tier {title}{comp_str()} — {yoe:.1f} years shipping {prod_str} with {skills_str}. {engage_str()}. {'Located in ' + loc_str + '.' if loc_str else ''}".strip(),
        ]
        
    elif rank <= 50:
        templates = [
            # T4: Direct factual, no filler
            f"{title}{comp_str()}. {yoe:.1f} years, {prod_str}, skilled in {skills_str}. {engage_str() or avail_str() or 'Solid track record.'}",
            
            # T5: Evidence-first, then role
            f"{skills_str} experience across {yoe:.1f} years of {prod_str}. {'Currently ' + title.lower() + comp_str() + '.' if comp_str() else ''} {avail_str()}.",
            
            # T6: Metrics-led
            f"{yoe:.1f} years in {prod_str}. {skills_str}. {'Response rate: ' + str(response) + '%.' if response > 60 else ''} {f'Located in {loc_str}.' if loc_str else ''}",
        ]
        
    else:
        templates = [
            # T7: Concern-forward when present
            (f"Concern: {concern_str}. " if concern_str else "") + f"{title} with {yoe:.1f} years in {prod_str}. {skills_str}. {avail_str() or 'Solid track record.'}",
            
            # T8: Profile summary, detached tone
            f"Profile spans {yoe:.1f} years of {prod_str}. Skills: {skills_str}. {'Response rate ' + str(response) + '%.' if response > 50 else ''}",
            
            # T9: Minimal, just the facts
            f"{yoe:.1f} years. {prod_str}. {skills_str}.{comp_str() + '.' if comp_str() else ''} {'Located in ' + loc_str + '.' if loc_str else ''}",
        ]
    
    template_idx = h % len(templates)
    result = templates[template_idx]
    
    # Clean up double spaces or trailing punctuation issues
    result = " ".join(result.split())
    result = result.strip()
    if result and result[-1] not in ".!":
        result += "."
        
    return result
