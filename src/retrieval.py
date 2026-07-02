import numpy as np
from src.config import CONFIG, WEIGHTS
from src.feature_compiler import F_VECTOR_DB, F_RETRIEVAL_RAG, F_EVAL, F_PYTHON, F_PROD_DEPTH, F_SHIPPED_SYS, F_INDUSTRY_FIT

def get_dual_lane_top_k(feature_matrix: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """
    Phase 7: Dual Lane Retrieval
    Instead of one funnel, we use two to protect hidden gems.
    Lane A: Top K based heavily on ML/AI skills.
    Lane B: Top K based heavily on Production/Shipped Systems.
    Returns the union of these indices, capped at TOP_K_SEMANTIC.
    """
    N = feature_matrix.shape[0]
    
    # Base Tech Score = Matrix @ Weights - Penalties
    # (We re-use this for the final tech score later, but we compute lane-specific scores here for retrieval)
    
    # Lane A: AI Heavy
    # Boost F0-F3
    ai_weights = np.array(WEIGHTS, dtype=np.float32)
    ai_weights[F_VECTOR_DB] *= 2.0
    ai_weights[F_RETRIEVAL_RAG] *= 2.0
    ai_weights[F_EVAL] *= 2.0
    ai_weights[F_PYTHON] *= 1.5
    ai_scores = (feature_matrix @ ai_weights) - penalties
    lane_a_indices = np.argsort(ai_scores)[::-1][:CONFIG["TOP_K_AI"]]
    
    # Lane B: Production Heavy
    # Boost F7, F13, F14
    prod_weights = np.array(WEIGHTS, dtype=np.float32)
    prod_weights[F_PROD_DEPTH] *= 3.0
    prod_weights[F_SHIPPED_SYS] *= 3.0
    prod_weights[F_INDUSTRY_FIT] *= 2.0
    prod_scores = (feature_matrix @ prod_weights) - penalties
    lane_b_indices = np.argsort(prod_scores)[::-1][:CONFIG["TOP_K_PRODUCTION"]]
    
    # Union
    union_set = set(lane_a_indices).union(set(lane_b_indices))
    union_indices = list(union_set)
    
    # If the union is larger than semantic limit, sort them by pure base technical score and trim
    if len(union_indices) > CONFIG["TOP_K_SEMANTIC"]:
        base_scores = (feature_matrix @ np.array(WEIGHTS, dtype=np.float32)) - penalties
        # Sort union by base_score descending
        union_indices.sort(key=lambda i: base_scores[i], reverse=True)
        union_indices = union_indices[:CONFIG["TOP_K_SEMANTIC"]]
        
    return np.array(union_indices, dtype=np.int32)
