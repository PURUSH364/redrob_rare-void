import gzip
import json
import os


def load_candidates(filepath):
    """Load candidates from a JSONL or JSONL.GZ file."""
    opener = gzip.open if filepath.endswith('.gz') else open
    candidates = []
    with opener(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def resolve_data_path():
    """Resolve the candidates data path relative to the analysis directory."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'candidates.jsonl')
    if not os.path.exists(data_path):
        data_path_gz = data_path + '.gz'
        if os.path.exists(data_path_gz):
            return data_path_gz
    return data_path
