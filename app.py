import gradio as gr
import tempfile
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

def rank_candidates(input_file):
    """Run the pipeline in-process and return ranked CSV."""
    if input_file is None:
        sample_path = os.path.join(PROJECT_ROOT, "small_sample.jsonl")
        if os.path.exists(sample_path):
            input_file = sample_path
        else:
            return None, "Please upload a candidates JSONL file."

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "candidates.jsonl")
            output_path = os.path.join(tmpdir, "submission.csv")
            shutil.copy(input_file, input_path)

            evidence_list = []
            for cand in stream_candidates(input_path):
                evidence_list.append(extract_evidence(cand))

            feature_matrix = compile_feature_matrix(evidence_list)
            penalties = apply_penalties(evidence_list)
            top_indices = get_dual_lane_top_k(feature_matrix, penalties)

            top_evidence = [evidence_list[i] for i in top_indices]
            semantic_scores = compute_semantic_scores(top_evidence)
            behavior_multipliers = compute_behavior_multiplier(evidence_list)

            base_scores = (feature_matrix @ np.array(WEIGHTS, dtype=np.float32)) - penalties
            final_top_k = aggregate_scores(
                base_scores, semantic_scores, behavior_multipliers, penalties[top_indices],
                evidence_list, top_indices, target_k=CONFIG["TOP_K_FINAL"]
            )

            for cand in final_top_k:
                idx = cand["_idx"]
                cand["reasoning"] = generate_reasoning(evidence_list[idx], rank=cand.get("rank", 100))

            for i in range(1, len(final_top_k)):
                if final_top_k[i]['score'] > final_top_k[i - 1]['score']:
                    final_top_k[i]['score'] = final_top_k[i - 1]['score']

            write_submission(final_top_k, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            row_count = len(lines) - 1

            final_output = os.path.join(PROJECT_ROOT, "output", "submission.csv")
            os.makedirs(os.path.dirname(final_output), exist_ok=True)
            shutil.copy(output_path, final_output)

            return final_output, f"Success! Ranked {row_count} candidates. Download the CSV below."

    except Exception as e:
        import traceback
        return None, f"Error:\n{traceback.format_exc()}"

with gr.Blocks(title="RARE_VOID - Candidate Ranking Pipeline") as demo:
    gr.Markdown("""
    # RARE_VOID — 13-Phase Candidate Ranking Pipeline

    Upload a `candidates.jsonl` file.
    **If you leave this empty, the demo will automatically use a pre-loaded sample of 200 candidates.**
    The pipeline will rank candidates and produce a submission CSV.

    **Command to reproduce locally:**
    ```
    python -m src.main --candidates data/candidates.jsonl --out output/team_rare_void.csv
    ```
    """)

    with gr.Row():
        input_file = gr.File(label="Upload candidates.jsonl", file_types=[".jsonl", ".gz"])

    run_btn = gr.Button("Run Pipeline", variant="primary")

    output_file = gr.File(label="Ranked Submission CSV")
    output_msg = gr.Textbox(label="Status")

    run_btn.click(fn=rank_candidates, inputs=input_file, outputs=[output_file, output_msg])

if __name__ == "__main__":
    demo.launch()
