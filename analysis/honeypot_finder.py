import sys
import os
import csv
from analysis.utils import load_candidates, resolve_data_path


def main():
    data_path = resolve_data_path()
    if not os.path.exists(data_path):
        print("ERROR: candidates.jsonl not found")
        sys.exit(1)

    print(f"Loading {data_path}...")
    candidates = load_candidates(data_path)
    print(f"Loaded {len(candidates)} candidates\n")

    honeypots = []
    suspicious = []

    for c in candidates:
        p = c.get('profile', {})
        years_exp = p.get('years_of_experience', 0)
        career_history = c.get('career_history', [])
        total_months = sum(r.get('duration_months', 0) for r in career_history)
        calc_years = total_months / 12
        mismatch = abs(calc_years - years_exp)

        reasons = []

        if mismatch > 2.5:
            reasons.append(f"YEAR MISMATCH: claimed {years_exp}y but career history = {calc_years:.1f}y (delta={mismatch:.1f})")

        if mismatch > 1.5 and mismatch <= 2.5:
            reasons.append(f"SUSPICIOUS: year mismatch {mismatch:.1f} (claimed {years_exp}y, calculated {calc_years:.1f}y)")

        skills = c.get('skills', [])
        expert_skills = sum(1 for s in skills if s.get('proficiency') == 'expert')
        if expert_skills >= 5 and years_exp < 3:
            reasons.append(f"SKILL INFLATION: {expert_skills} expert skills with only {years_exp}y experience")

        if len(career_history) >= 5 and total_months < 60:
            reasons.append(f"JOB HOPPING: {len(career_history)} roles in {total_months} months")

        if reasons:
            entry = {
                'candidate_id': c['candidate_id'],
                'title': p.get('current_title', ''),
                'years_exp': years_exp,
                'calc_years': calc_years,
                'mismatch': mismatch,
                'reasons': reasons
            }
            if mismatch > 2.5:
                honeypots.append(entry)
            else:
                suspicious.append(entry)

    print("=" * 70)
    print("HONEYPOT & SUSPICIOUS PROFILE REPORT")
    print("=" * 70)

    print(f"\nDefinite Honeypots (mismatch > 2.5 years): {len(honeypots)}")
    print(f"Suspicious profiles (mismatch 1.5-2.5 years): {len(suspicious)}")
    print(f"Total flagged: {len(honeypots) + len(suspicious)}")

    if honeypots:
        print(f"\n--- TOP 30 DEFINITE HONEYPOTS ---")
        honeypots.sort(key=lambda x: x['mismatch'], reverse=True)
        for i, h in enumerate(honeypots[:30], 1):
            print(f"\n  {i}. {h['candidate_id']} - {h['title']}")
            print(f"     Claimed: {h['years_exp']}y | Calculated: {h['calc_years']:.1f}y | Delta: {h['mismatch']:.1f}y")
            for r in h['reasons']:
                print(f"     -> {r}")

    if suspicious:
        print(f"\n--- TOP 20 SUSPICIOUS PROFILES ---")
        suspicious.sort(key=lambda x: x['mismatch'], reverse=True)
        for i, s in enumerate(suspicious[:20], 1):
            print(f"\n  {i}. {s['candidate_id']} - {s['title']}")
            print(f"     Claimed: {s['years_exp']}y | Calculated: {s['calc_years']:.1f}y | Delta: {s['mismatch']:.1f}y")
            for r in s['reasons']:
                print(f"     -> {r}")

    # Export flagged candidates to CSV
    report_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'honeypot_report.csv')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    all_flagged = honeypots + suspicious
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'title', 'years_exp', 'calc_years', 'mismatch', 'reasons'])
        for entry in all_flagged:
            writer.writerow([
                entry['candidate_id'], entry['title'], entry['years_exp'],
                f"{entry['calc_years']:.1f}", f"{entry['mismatch']:.1f}",
                '; '.join(entry['reasons'])
            ])
    print(f"\nReport exported to {report_path}")


if __name__ == '__main__':
    main()
