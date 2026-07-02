import gradio as gr
import tempfile
import os
import subprocess
import sys

def rank_candidates(input_file):
    """Run the pipeline on uploaded candidates JSONL and return ranked CSV."""
    if input_file is None:
        return None, "Please upload a candidates JSONL file."
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "candidates.jsonl")
        output_path = os.path.join(tmpdir, "submission.csv")
        
        # Copy uploaded file
        import shutil
        shutil.copy(input_file, input_path)
        
        # Run pipeline
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--candidates", input_path, "--out", output_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return None, f"Error: {result.stderr}"
        
        if not os.path.exists(output_path):
            return None, "Pipeline did not produce output."
        
        # Count rows
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        row_count = len(lines) - 1  # minus header
        
        return output_path, f"Success! Ranked {row_count} candidates. Download the CSV below."
    
with gr.Blocks(title="RARE_VOID - Candidate Ranking Pipeline") as demo:
    gr.Markdown("""
    # RARE_VOID — 13-Phase Candidate Ranking Pipeline
    
    Upload a `candidates.jsonl` file (≤100 candidates recommended for this demo).
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
