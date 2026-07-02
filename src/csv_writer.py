import csv
import os


def write_submission(results: list, output_path: str) -> str:
    """
    Write ranked results to a UTF-8 CSV.
    Expects each result dict to have: candidate_id, rank, score, reasoning.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in results:
            writer.writerow([
                r['candidate_id'],
                r['rank'],
                f"{r['score']:.6f}",
                r.get('reasoning', ''),
            ])
    return output_path
