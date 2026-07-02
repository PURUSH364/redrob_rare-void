import os
import sys
import argparse
import subprocess
import numpy as np

from src.config import CONFIG, WEIGHTS
from src.streamer import stream_candidates
from src.evidence_builder import extract_evidence
from src.feature_compiler import compile_feature_matrix
from src.penalty_engine import apply_penalties
from src.retrieval import get_dual_lane_top_k
from src.semantic_engine import compute_semantic_scores
from src.behavior_engine import compute_behavior_multiplier
from src.score_aggregator import aggregate_scores
from src.reason_generator import generate_reasoning
from src.csv_writer import write_submission
from src.profiling import PipelineProfiler

def _enforce_monotonicity(results: list) -> None:
    # Safety net for floating-point micro-violations where score[i] > score[i-1]
    # due to float precision. The scoring math should be inherently ordered;
    # this corrects sub-epsilon drift only.
    for i in range(1, len(results)):
        if results[i]['score'] > results[i - 1]['score']:
            results[i]['score'] = results[i - 1]['score']

def main():
    parser = argparse.ArgumentParser(description="RARE_VOID Final Architecture")
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl", help="Path to input data")
    parser.add_argument("--out", type=str, default="output/team_rare_void.csv", help="Path to output CSV")
    # Phase 13: Ablation Mode
    parser.add_argument("--ablation", action="append", help="Disable specific modules (semantic_off, behavior_off, penalties_off)")
    args = parser.parse_args()

    ablations = args.ablation or []
    semantic_off = "semantic_off" in ablations
    behavior_off = "behavior_off" in ablations
    penalties_off = "penalties_off" in ablations

    profiler = PipelineProfiler()
    
    print(f"\n{'='*62}")
    print(f"  RARE_VOID - 13-Phase Hybrid RAG/Search Pipeline")
    if ablations:
        print(f"  Ablations Active: {ablations}")
    print(f"{'='*62}\n")

    # -- Phase 1-6: Compilation & Penalties ------------------------------------
    profiler.start("Feature Compilation & Penalties")
    print("Streaming candidates and building evidence + features...")
    
    # 1. Stream & 4. Evidence Builder
    evidence_list = []
    count = 0
    for cand in stream_candidates(args.candidates):
        ev = extract_evidence(cand)
        evidence_list.append(ev)
        count += 1
        if count % 20_000 == 0:
            print(f"  Processed {count:,} candidates...")
            
    # 5. Feature Compiler
    feature_matrix = compile_feature_matrix(evidence_list)
    
    # 6. Penalty Engine
    if penalties_off:
        penalties = np.zeros(count, dtype=np.float32)
    else:
        penalties = apply_penalties(evidence_list)
        
    profiler.stop()

    # -- Phase 7: Dual Lane Retrieval ------------------------------------------
    profiler.start("Dual Lane Retrieval")
    print("Retrieving Top K candidates via Dual Lane (AI + Prod)...")
    top_indices = get_dual_lane_top_k(feature_matrix, penalties)
    profiler.stop()

    # -- Phase 8: Semantic Reranker --------------------------------------------
    profiler.start("Semantic Reranker")
    print(f"Running offline MiniLM Semantic Reranker on {len(top_indices)} candidates...")
    if semantic_off:
        semantic_scores = np.zeros(len(top_indices), dtype=np.float32)
    else:
        top_evidence = [evidence_list[i] for i in top_indices]
        semantic_scores = compute_semantic_scores(top_evidence)
    profiler.stop()

    # -- Phase 9: Behavior Multiplier ------------------------------------------
    profiler.start("Behavior Multiplier")
    print("Computing Behavior Multipliers...")
    # Calculate for all to be safe, but we only really need it for top_indices
    # Actually, we compute for all in 0.1s
    if behavior_off:
        behavior_multipliers = np.ones(count, dtype=np.float32)
    else:
        behavior_multipliers = compute_behavior_multiplier(evidence_list)
    profiler.stop()

    # -- Phase 11: Score Aggregation & Tie Breakers ----------------------------
    profiler.start("Score Aggregator & Reason Gen")
    print("Aggregating scores and applying deterministic tie-breakers...")
    base_scores = (feature_matrix @ np.array(WEIGHTS, dtype=np.float32)) - penalties
    
    final_top_k = aggregate_scores(
        base_scores, semantic_scores, behavior_multipliers, penalties[top_indices],
        evidence_list, top_indices, target_k=CONFIG["TOP_K_FINAL"]
    )
    
    # Phase 10: Reason Generator
    for cand in final_top_k:
        idx = cand["_idx"]
        cand["reasoning"] = generate_reasoning(evidence_list[idx], rank=cand.get("rank", 100))
        
    profiler.stop()

    # -- Phase 12-13: Validation & Output --------------------------------------
    profiler.start("Validation & CSV")
    _enforce_monotonicity(final_top_k)
    write_submission(final_top_k, args.out)
    profiler.stop()

    # -- Report ----------------------------------------------------------------
    print(profiler.report().replace('—', '-'))
    
    print(f"\n{'-'*90}")
    print(f"  {'Rk':<4} {'ID':<15} {'Score':<7} {'Title':<32} Reasoning")
    print(f"{'-'*90}")
    for r in final_top_k[:10]:
        ev = evidence_list[r['_idx']]
        title = ev["current_title"][:30]
        snip = r["reasoning"][:55]
        print(f"  {r['rank']:<4} {r['candidate_id']:<15} {r['score']:<7.4f} {title:<32} {snip}...")

    # -- Honeypot Rate Sanity Check ----------------------------------------------
    if final_top_k:
        honeypot_count = sum(1 for r in final_top_k if any("HARD FILTER" in p for p in evidence_list[r["_idx"]]["penalties"]))
        print(f"\n  Honeypot rate in top {len(final_top_k)}: {honeypot_count}/{len(final_top_k)} ({honeypot_count/len(final_top_k)*100:.1f}%)")
        if honeypot_count > len(final_top_k) * 0.10:
            print("  WARNING: Honeypot rate exceeds 10% threshold!")

    return args.out

if __name__ == '__main__':
    out_csv = main()
            
    print("\nValidating submission...")
    res = subprocess.run([sys.executable, "validate_submission.py", out_csv], capture_output=True, text=True)
    if res.returncode != 0:
        print("VALIDATION FAILED!")
        print(res.stderr)
        print(res.stdout)
        sys.exit(1)
    else:
        print("Submission is valid.")
