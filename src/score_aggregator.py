import numpy as np
from src.config import CONFIG

def aggregate_scores(
    base_scores: np.ndarray,
    semantic_scores: np.ndarray,
    behavior_multipliers: np.ndarray,
    penalties_subset: np.ndarray,
    evidence_list: list,
    top_indices: np.ndarray,
    target_k: int = 100
) -> list:
    """
    Phase 11: Final Aggregator & Tie Breakers
    Feature scores are already bounded [0, 1] (weights sum to 1.0, features in [0, 1]).
    Semantic scores are already normalized [0, 1] upstream (10th/90th percentile anchoring).
    Combines them (0.60 feature / 0.40 semantic) without re-normalization.
    Applies behavioral multiplier.
    Sorts and applies deterministic tie-breakers.
    Returns list of dicts for the final Top 100.
    """
    # 1. Slice the arrays for the top_indices subset
    feat_subset = base_scores[top_indices]
    sem_subset = semantic_scores  # Already normalized [0, 1] by semantic_engine.py
    beh_subset = behavior_multipliers[top_indices]
    
    # 2. Clip feature scores to [0, 1] as a safety net (penalties can push below 0)
    feat_clipped = np.clip(feat_subset, 0.0, 1.0)
    
    # 3. Combine Technical Score (0.60 feature / 0.40 semantic)
    w_f = CONFIG["FEATURE_WEIGHT"]
    w_s = CONFIG["SEMANTIC_WEIGHT"]
    final_technical = (w_f * feat_clipped) + (w_s * sem_subset)
    
    # 4. Apply Behavior Multiplier
    final_total = np.clip(final_technical * beh_subset, 0.0, 1.0)
    
    # 5. Build sortable list with tie-breakers
    # Priority: Final Score (desc), Response Rate (desc), Years Exp (desc), Github (desc), ID (asc)
    ranked_candidates = []
    for i, original_idx in enumerate(top_indices):
        # Drop hard filtered candidates entirely
        if penalties_subset[i] >= CONFIG["MAX_PENALTY"]:
            continue
            
        ev = evidence_list[original_idx]
        score = float(final_total[i])
        
        # Tie breakers
        response = ev["response_rate"]
        yoe = ev["years_exp"]
        github = ev["github_score"]
        cid = ev["id"]
        
        ranked_candidates.append({
            "_idx": original_idx,
            "candidate_id": cid,
            "score": score,
            "_feat_score": float(feat_clipped[i]),
            "_sem_score": float(sem_subset[i]),
            "_beh_mult": float(beh_subset[i]),
            "_tie_response": response,
            "_tie_yoe": yoe,
            "_tie_github": github
        })
        
    # Tie-breaker: score desc, then candidate_id ascending (validator requirement)
    ranked_candidates.sort(
        key=lambda x: (
            -x["score"],
            x["candidate_id"]
        ),
        reverse=False
    )
    
    # Slice to final Top K
    final_top = ranked_candidates[:target_k]
    
    # Assign integer ranks
    for rank, cand in enumerate(final_top, start=1):
        cand["rank"] = rank
        
    return final_top
