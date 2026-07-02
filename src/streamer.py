import json
import gzip

def stream_candidates(filepath, max_candidates=None):
    """
    Phase 1: Streaming Parser
    Pure generator. Opens the file, yields JSON dicts one by one.
    Do NOT load everything into memory.
    """
    opener = gzip.open if filepath.endswith('.gz') else open
    count = 0
    with opener(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield candidate
                count += 1
                if max_candidates and count >= max_candidates:
                    break
            except json.JSONDecodeError:
                continue
