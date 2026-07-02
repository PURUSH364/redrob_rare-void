import os
import numpy as np
import warnings

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

def compute_semantic_scores(top_evidence: list) -> np.ndarray:
    """
    Phase 8: Semantic Reranker
    Uses local MiniLM. No network dependencies.
    Normalises semantic score 0 -> 1.
    """
    # Check if the library is available
    if not _HAS_SENTENCE_TRANSFORMERS:
        print("  WARNING: sentence-transformers not installed! Semantic scores will be 0.")
        return np.zeros(len(top_evidence), dtype=np.float32)

    # Load from the local path explicitly
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_model")
    
    # If the model directory doesn't exist, this means setup_model.py wasn't run or failed.
    # We should fall back gracefully to 0 scores to allow the pipeline to proceed.
    if not os.path.exists(model_path):
        print("  WARNING: ./local_model not found! Semantic scores will be 0. Run setup_model.py.")
        return np.zeros(len(top_evidence), dtype=np.float32)
        
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        print(f"  WARNING: Failed to load local model: {e}")
        return np.zeros(len(top_evidence), dtype=np.float32)

    # Core JD to match against
    jd_text = (
        "Senior AI/Machine Learning Engineer. "
        "Production experience with Vector Databases (Pinecone, Milvus, Qdrant). "
        "Building Retrieval Augmented Generation (RAG) and Semantic Search pipelines. "
        "Evaluating LLM and search systems using NDCG, MRR, RAGAS, or Truera. "
        "Python, FastAPI, Recommendation Systems, Learning to Rank (LTR), and fine-tuning. "
        "Building and shipping AI systems at scale."
    )
    
    jd_emb = model.encode([jd_text], convert_to_tensor=True, show_progress_bar=False)
    
    # Build a descriptive text payload for each candidate
    candidate_texts = []
    for ev in top_evidence:
        # Build a text block that semantically describes them
        title = ev["current_title"]
        skills = ", ".join([s["name"] for s in ev["top_skills"][:10]])
        career = " ".join([snip.get("snippet", "") for snip in ev["career_snippets"]])
        text = f"Title: {title}. Skills: {skills}. Experience: {career}"
        candidate_texts.append(text)
        
    cand_embs = model.encode(candidate_texts, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute cosine similarity
    similarities = cos_sim(jd_emb, cand_embs).cpu().numpy().flatten()
    
    # Percentile-based normalization (robust to outliers)
    # Uses 10th/90th percentiles as anchors instead of min/max to prevent
    # single outliers from compressing the entire score distribution
    p10 = np.percentile(similarities, 10)
    p90 = np.percentile(similarities, 90)
    if p90 > p10:
        semantic_norm = np.clip((similarities - p10) / (p90 - p10), 0.0, 1.0)
    else:
        semantic_norm = np.zeros_like(similarities)
        
    return semantic_norm.astype(np.float32)
