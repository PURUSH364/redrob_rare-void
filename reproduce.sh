#!/bin/bash
# reproduce.sh - End-to-end pipeline for India Data Challenge

set -e

# 0. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 0b. Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q pytest

echo "Starting RARE_VOID pipeline..."

# 1. Run unit tests
python -m pytest tests/test_pipeline.py -v

# 2. Run the pipeline end-to-end (pre-processing, scoring, ranking)
python -m src.main --candidates data/candidates.jsonl --out output/team_rare_void.csv

# 3. Validate the final submission format
python validate_submission.py output/team_rare_void.csv

echo "Pipeline completed successfully!"
