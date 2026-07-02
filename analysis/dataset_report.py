import sys
import os
import csv
from collections import Counter, defaultdict
from analysis.utils import load_candidates, resolve_data_path


def main():
    data_path = resolve_data_path()
    if not os.path.exists(data_path):
        print("ERROR: candidates.jsonl not found")
        sys.exit(1)

    print(f"Loading {data_path}...")
    candidates = load_candidates(data_path)
    print(f"Loaded {len(candidates)} candidates\n")

    title_counts = Counter()
    industry_counts = Counter()
    company_counts = Counter()
    country_counts = Counter()

    response_rates = []
    github_scores = []
    notice_days = []
    years_exp = []
    career_months = []
    open_to_work_count = 0
    consulting_count = 0

    CONSULTING = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
                  'tech mahindra', 'hcl', 'hcltech', 'ltimindtree', 'mindtree'}

    for c in candidates:
        p = c.get('profile', {})
        title_counts[p.get('current_title', 'UNKNOWN')] += 1
        industry_counts[p.get('current_industry', 'UNKNOWN')] += 1
        company_counts[p.get('current_company', 'UNKNOWN')] += 1
        country_counts[p.get('country', 'UNKNOWN')] += 1

        sig = c.get('redrob_signals', {})
        response_rates.append(sig.get('recruiter_response_rate', 0))
        github_scores.append(sig.get('github_activity_score', -1))
        notice_days.append(sig.get('notice_period_days', 0))
        years_exp.append(p.get('years_of_experience', 0))
        open_to_work_count += sig.get('open_to_work_flag', False)

        total_months = sum(r.get('duration_months', 0) for r in c.get('career_history', []))
        career_months.append(total_months)

        company_lower = p.get('current_company', '').lower()
        if any(firm in company_lower for firm in CONSULTING):
            consulting_count += 1

    print("=" * 60)
    print("DATASET REPORT")
    print("=" * 60)

    print(f"\nTotal candidates: {len(candidates)}")
    if len(candidates) == 0:
        print("No candidates found in dataset.")
        return
    print(f"Open to work: {open_to_work_count} ({open_to_work_count/len(candidates)*100:.1f}%)")
    print(f"Currently at consulting firm: {consulting_count} ({consulting_count/len(candidates)*100:.1f}%)")

    print(f"\n--- Years of Experience ---")
    years_exp.sort()
    print(f"  Min: {years_exp[0]}, Max: {years_exp[-1]}")
    print(f"  Median: {years_exp[len(years_exp)//2]}")
    print(f"  Mean: {sum(years_exp)/len(years_exp):.1f}")

    print(f"\n--- Recruiter Response Rate ---")
    response_rates.sort()
    print(f"  Min: {response_rates[0]:.2f}, Max: {response_rates[-1]:.2f}")
    print(f"  Median: {response_rates[len(response_rates)//2]:.2f}")
    print(f"  Mean: {sum(response_rates)/len(response_rates):.2f}")

    print(f"\n--- GitHub Activity Score ---")
    gh_positive = [g for g in github_scores if g >= 0]
    print(f"  Has GitHub linked: {len(gh_positive)} ({len(gh_positive)/len(candidates)*100:.1f}%)")
    if gh_positive:
        gh_positive.sort()
        print(f"  Median (among linked): {gh_positive[len(gh_positive)//2]}")
        print(f"  Mean (among linked): {sum(gh_positive)/len(gh_positive):.1f}")

    print(f"\n--- Notice Period (days) ---")
    notice_days.sort()
    print(f"  Median: {notice_days[len(notice_days)//2]}")
    print(f"  <=30 days: {sum(1 for n in notice_days if n <= 30)} ({sum(1 for n in notice_days if n <= 30)/len(candidates)*100:.1f}%)")

    print(f"\n--- Top 20 Titles ---")
    for title, count in title_counts.most_common(20):
        print(f"  {title:<40} {count:>5} ({count/len(candidates)*100:.1f}%)")

    print(f"\n--- Top 15 Industries ---")
    for ind, count in industry_counts.most_common(15):
        print(f"  {ind:<40} {count:>5} ({count/len(candidates)*100:.1f}%)")

    print(f"\n--- Top 15 Countries ---")
    for country, count in country_counts.most_common(15):
        print(f"  {country:<40} {count:>5} ({count/len(candidates)*100:.1f}%)")

    # Export summary to CSV
    report_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'dataset_report.csv')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_candidates', len(candidates)])
        writer.writerow(['open_to_work', open_to_work_count])
        writer.writerow(['consulting_count', consulting_count])
        writer.writerow(['median_yoe', years_exp[len(years_exp)//2]])
        writer.writerow(['mean_yoe', f'{sum(years_exp)/len(years_exp):.1f}'])
        writer.writerow(['median_response_rate', f'{response_rates[len(response_rates)//2]:.2f}'])
        for title, count in title_counts.most_common(20):
            writer.writerow([f'title:{title}', count])
    print(f"\nReport exported to {report_path}")


if __name__ == '__main__':
    main()
