import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.config import CONFIG, WEIGHTS, NUM_FEATURES, CONSULTING_FIRMS, LD_KEYWORDS
from src.score_aggregator import aggregate_scores
from src.behavior_engine import compute_behavior_multiplier
from src.penalty_engine import apply_penalties
from src.feature_compiler import compile_feature_matrix
from src.reason_generator import generate_reasoning


def _ev(**kw):
    base = dict(
        id="CAND_001", years_exp=7.0, current_title="Senior ML Engineer",
        location="Bangalore, Karnataka", country="India", response_rate=0.80,
        open_to_work=True, notice_days=60, saved_30d=5,
        avg_response_time_hours=12, profile_completeness_score=85,
        interview_completion_rate=0.70, applications_30d=3, github_score=75.0,
        total_career_months=84, willing_to_relocate=False,
        preferred_work_mode="hybrid", expected_salary_min=25,
        expected_salary_max=40, last_active_date="2026-06-01",
        top_skills=[
            {"name": "faiss", "proficiency": "expert", "duration_months": 24},
            {"name": "pinecone", "proficiency": "advanced", "duration_months": 12},
        ],
        education=[{"field_of_study": "Computer Science", "tier": "tier_1", "degree": "b.tech"}],
        career_history=[{"company": "Flipkart", "title": "ML Engineer",
            "description": "built ranking with faiss", "start_date": "2020-01-01"}],
        career_snippets=[{"company": "Flipkart", "title": "ML Engineer"}],
        vector_db_found=["faiss"], retrieval_found=["rag"], eval_found=["ndcg"],
        python_found=["python"], production_found=["faiss"], penalties=[],
        jd_match_count=3,
        _source_candidate={"career_history": [{"company": "Flipkart",
            "title": "ML Engineer", "description": "built ranking with faiss",
            "start_date": "2020-01-01"}]},
    )
    base.update(kw)
    return base


# === Config ===

def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS) - 1.0) < 1e-6

def test_num_features():
    assert NUM_FEATURES == 20

def test_behavior_bounds():
    assert 0.0 < CONFIG["BEHAVIOR_MIN"] < CONFIG["BEHAVIOR_MAX"] <= 1.5

def test_max_penalty_positive():
    assert CONFIG["MAX_PENALTY"] > 0


# === Score Aggregator ===

def test_hard_filter_drops_candidate():
    e1 = _ev(id="CAND_A", penalties=["-0.50 (HARD FILTER): Consulting"])
    e2 = _ev(id="CAND_B")
    r = aggregate_scores(
        np.array([0.9, 0.5]), np.array([0.8, 0.6]),
        np.array([1.0, 1.0]), np.array([0.50, 0.0]),
        [e1, e2], np.array([0, 1]), target_k=10)
    ids = [c["candidate_id"] for c in r]
    assert "CAND_A" not in ids
    assert "CAND_B" in ids

def test_tiebreaker_id_ascending():
    e1 = _ev(id="CAND_ZZZ", penalties=[])
    e2 = _ev(id="CAND_AAA", penalties=[])
    r = aggregate_scores(
        np.array([0.9, 0.9]), np.array([0.8, 0.8]),
        np.array([1.0, 1.0]), np.array([0.0, 0.0]),
        [e1, e2], np.array([0, 1]), target_k=10)
    assert r[0]["candidate_id"] == "CAND_AAA"

def test_output_has_rank():
    e = _ev(penalties=[])
    r = aggregate_scores(
        np.array([0.7]), np.array([0.6]),
        np.array([1.0]), np.array([0.0]),
        [e], np.array([0]), target_k=10)
    assert r[0]["rank"] == 1
    assert "score" in r[0]


# === Behavior Engine ===

def test_multiplier_in_bounds():
    m = compute_behavior_multiplier([_ev()])[0]
    assert CONFIG["BEHAVIOR_MIN"] <= m <= CONFIG["BEHAVIOR_MAX"]

def test_high_response_beats_low():
    h = compute_behavior_multiplier([_ev(id="H", response_rate=0.95)])[0]
    l = compute_behavior_multiplier([_ev(id="L", response_rate=0.10)])[0]
    assert h > l

def test_inactive_penalized():
    a = compute_behavior_multiplier([_ev(id="A", last_active_date="2026-06-01")])[0]
    i = compute_behavior_multiplier([_ev(id="B", last_active_date="2025-01-01")])[0]
    assert a > i

def test_all_zeros_gives_min():
    e = _ev(id="Z", response_rate=0.0, github_score=-1, open_to_work=False,
            notice_days=180, saved_30d=0, avg_response_time_hours=200,
            profile_completeness_score=10, interview_completion_rate=0.0,
            applications_30d=0, last_active_date="")
    m = compute_behavior_multiplier([e])[0]
    assert m == CONFIG["BEHAVIOR_MIN"]


# === Penalty Engine ===

def test_consulting_only_hard_filter():
    career = [{"company": "Wipro", "title": "Consultant",
        "description": "client work", "start_date": "2018-01-01"}]
    e = _ev(id="C1", current_title="Consultant", career_history=career,
            _source_candidate={"career_history": career}, production_found=[])
    p = apply_penalties([e])
    assert p[0] >= 0.50

def test_cv_no_retrieval_hard_filter():
    e = _ev(id="C2", current_title="Computer Vision Engineer",
            retrieval_found=[],
            _source_candidate={"career_history": [
                {"company": "Lab", "title": "CV Researcher",
                 "description": "image classification", "start_date": "2021-01-01"}]})
    p = apply_penalties([e])
    assert p[0] >= 0.50

def test_expert_zero_months_penalized():
    e = _ev(id="C3", top_skills=[
        {"name": "faiss", "proficiency": "expert", "duration_months": 0},
        {"name": "pinecone", "proficiency": "expert", "duration_months": 0},
        {"name": "weaviate", "proficiency": "expert", "duration_months": 0}])
    p = apply_penalties([e])
    assert p[0] >= 0.20

def test_penalty_capped():
    career = [{"company": "Wipro", "title": "Consultant",
        "description": "work", "start_date": "2018-01-01"}]
    e = _ev(id="C4", current_title="Consultant", years_exp=5.0,
            total_career_months=10, career_history=career,
            _source_candidate={"career_history": career}, production_found=[],
            top_skills=[
                {"name": "a", "proficiency": "expert", "duration_months": 0},
                {"name": "b", "proficiency": "expert", "duration_months": 0},
                {"name": "c", "proficiency": "expert", "duration_months": 0}])
    p = apply_penalties([e])
    assert p[0] <= CONFIG["MAX_PENALTY"]

def test_notice_penalty():
    e = _ev(id="C5", notice_days=150)
    p = apply_penalties([e])
    assert p[0] >= 0.05


# === Feature Compiler ===

def test_features_in_0_1():
    m = compile_feature_matrix([_ev()])
    assert m.shape == (1, NUM_FEATURES)
    assert np.all(m >= 0.0) and np.all(m <= 1.0)

def test_keyword_stuffing_defense():
    good = _ev(id="G", top_skills=[
        {"name": "faiss", "proficiency": "expert", "duration_months": 24}])
    bad = _ev(id="B", years_exp=2.0, top_skills=[
        {"name": "faiss", "proficiency": "expert", "duration_months": 0},
        {"name": "pinecone", "proficiency": "expert", "duration_months": 0},
        {"name": "weaviate", "proficiency": "expert", "duration_months": 0},
        {"name": "milvus", "proficiency": "expert", "duration_months": 0},
        {"name": "qdrant", "proficiency": "expert", "duration_months": 0}])
    mg = compile_feature_matrix([good])
    mb = compile_feature_matrix([bad])
    assert mg[0, 0] >= mb[0, 0]

def test_location_fit_tier1():
    e1 = _ev(id="L1", location="pune, Maharashtra")
    e2 = _ev(id="L2", location="Rural Village")
    m1 = compile_feature_matrix([e1])
    m2 = compile_feature_matrix([e2])
    assert m1[0, 15] > m2[0, 15]

def test_pre_llm_feature():
    e_pre = _ev(id="P",
        _source_candidate={"career_history": [
            {"company": "X", "title": "ML Engineer",
             "description": "machine learning models", "start_date": "2019-01-01"}]})
    e_no = _ev(id="N",
        _source_candidate={"career_history": [
            {"company": "Y", "title": "Engineer",
             "description": "web dev", "start_date": "2023-01-01"}]})
    m1 = compile_feature_matrix([e_pre])
    m2 = compile_feature_matrix([e_no])
    assert m1[0, 19] > m2[0, 19]


# === Reason Generator ===

def test_reasoning_top10_confident():
    e = _ev()
    r = generate_reasoning(e, rank=5)
    assert "years" in r and "production" in r.lower() or "ML" in r

def test_reasoning_rank100_cautious():
    e = _ev()
    r = generate_reasoning(e, rank=95)
    assert "years" in r

def test_reasoning_deterministic():
    e = _ev()
    r1 = generate_reasoning(e, rank=50)
    r2 = generate_reasoning(e, rank=50)
    assert r1 == r2

def test_reasoning_varies_by_rank():
    e = _ev()
    r_top = generate_reasoning(e, rank=5)
    r_mid = generate_reasoning(e, rank=30)
    r_low = generate_reasoning(e, rank=80)
    assert not (r_top == r_mid == r_low)

def test_reasoning_includes_company():
    e = _ev()
    r = generate_reasoning(e, rank=5)
    assert "Flipkart" in r

def test_reasoning_includes_skills():
    e = _ev()
    r = generate_reasoning(e, rank=50)
    assert "Faiss" in r or "RAG" in r

def test_reasoning_structural_variation():
    e1 = _ev(id="CAND_ALPHA")
    e2 = _ev(id="CAND_BRAVO")
    e3 = _ev(id="CAND_CHARLIE")
    r1 = generate_reasoning(e1, rank=5)
    r2 = generate_reasoning(e2, rank=5)
    r3 = generate_reasoning(e3, rank=5)
    # All three should be different structures
    assert not (r1 == r2 == r3)


# === Improved Feature Tests ===

def test_product_co_ratio_mixed_career():
    """F8: A candidate with mixed consulting/product career should get an intermediate score."""
    e = _ev(id="MIX", total_career_months=120,
        _source_candidate={"career_history": [
            {"company": "Flipkart", "title": "ML Engineer",
             "description": "built ranking with faiss", "start_date": "2020-01-01",
             "duration_months": 60},
            {"company": "TCS", "title": "Consultant",
             "description": "client work", "start_date": "2015-01-01",
             "duration_months": 60},
        ]})
    m = compile_feature_matrix([e])
    # Should be ~0.5 (half consulting, half product)
    assert 0.3 <= m[0, 8] <= 0.7

def test_product_co_ratio_pure_product():
    """F8: A candidate with no consulting should get ~1.0."""
    e = _ev(id="PROD", total_career_months=60,
        _source_candidate={"career_history": [
            {"company": "Google", "title": "ML Engineer",
             "description": "built ranking", "start_date": "2020-01-01",
             "duration_months": 60},
        ]})
    m = compile_feature_matrix([e])
    assert m[0, 8] >= 0.9

def test_recency_recent_beats_stale():
    """F11: A recently active candidate should score higher than a stale one."""
    recent = _ev(id="R", last_active_date="2026-06-15")
    stale = _ev(id="S", last_active_date="2025-06-01")
    m_r = compile_feature_matrix([recent])
    m_s = compile_feature_matrix([stale])
    assert m_r[0, 11] > m_s[0, 11]

def test_recency_no_date_gets_neutral():
    """F11: Missing last_active_date should get 0.5 (neutral)."""
    e = _ev(id="N", last_active_date="")
    m = compile_feature_matrix([e])
    assert m[0, 11] == 0.5

def test_industry_fit_relevant():
    """F14: Career mentioning relevant industries should score higher."""
    relevant = _ev(id="IND",
        _source_candidate={"career_history": [
            {"company": "Flipkart", "title": "ML Engineer",
             "description": "e-commerce search ranking and marketplace optimization",
             "start_date": "2020-01-01"}]})
    generic = _ev(id="GEN",
        _source_candidate={"career_history": [
            {"company": "X Corp", "title": "Engineer",
             "description": "built internal tools",
             "start_date": "2020-01-01"}]})
    m_r = compile_feature_matrix([relevant])
    m_g = compile_feature_matrix([generic])
    assert m_r[0, 14] > m_g[0, 14]

def test_research_with_production_not_filtered():
    """Research title with production keywords in career text should NOT be hard filtered."""
    e = _ev(id="RS", current_title="Research Scientist",
        production_found=[],
        _source_candidate={"career_history": [
            {"company": "Google", "title": "Research Scientist",
             "description": "deployed production ML ranking models at scale",
             "start_date": "2019-01-01"}]})
    p = apply_penalties([e])
    # Should NOT get the pure_research hard filter
    assert p[0] < 0.50


# === Evidence Builder ===

def _full_candidate(**kw):
    """Build a full raw candidate dict for testing extract_evidence."""
    base = {
        "candidate_id": "CAND_001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "years_of_experience": 7.0,
            "location": "Bangalore, Karnataka",
            "country": "India",
            "headline": "ML Engineer at Flipkart",
        },
        "skills": [
            {"name": "faiss", "proficiency": "expert", "duration_months": 24, "endorsements": 5},
            {"name": "pinecone", "proficiency": "advanced", "duration_months": 12, "endorsements": 3},
        ],
        "career_history": [
            {"company": "Flipkart", "title": "ML Engineer",
             "description": "built ranking with faiss and rag pipeline",
             "start_date": "2020-01-01", "duration_months": 60, "is_current": True}
        ],
        "education": [
            {"field_of_study": "Computer Science", "tier": "tier_1", "degree": "b.tech"}
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.80,
            "github_activity_score": 75.0,
            "notice_period_days": 60,
            "open_to_work_flag": True,
            "saved_by_recruiters_30d": 5,
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
            "avg_response_time_hours": 12,
            "profile_completeness_score": 85,
            "interview_completion_rate": 0.70,
            "applications_submitted_30d": 3,
            "last_active_date": "2026-06-01",
        }
    }
    base.update(kw)
    return base


def test_extract_evidence_basic():
    """extract_evidence returns correct structure with all required fields."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert ev["id"] == "CAND_001"
    assert ev["years_exp"] == 7.0
    assert ev["current_title"] == "Senior ML Engineer"
    assert ev["response_rate"] == 0.80
    assert ev["github_score"] == 75.0
    assert ev["notice_days"] == 60
    assert ev["open_to_work"] is True
    assert ev["total_career_months"] == 60


def test_extract_evidence_vector_db_found():
    """extract_evidence detects vector DB skills in skill names."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert "faiss" in ev["vector_db_found"]


def test_extract_evidence_retrieval_found():
    """extract_evidence detects retrieval/RAG keywords in career text."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert len(ev["retrieval_found"]) > 0


def test_extract_evidence_empty_candidate():
    """extract_evidence handles candidate with no skills, career, or signals."""
    from src.evidence_builder import extract_evidence
    cand = {
        "candidate_id": "CAND_EMPTY",
        "profile": {},
        "skills": [],
        "career_history": [],
        "education": [],
        "redrob_signals": {}
    }
    ev = extract_evidence(cand)
    assert ev["id"] == "CAND_EMPTY"
    assert ev["years_exp"] == 0.0
    assert ev["vector_db_found"] == []
    assert ev["top_skills"] == []
    assert ev["penalties"] == []


def test_extract_evidence_career_snippets():
    """extract_evidence builds career_snippets from career_history."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert len(ev["career_snippets"]) == 1
    assert ev["career_snippets"][0]["company"] == "Flipkart"
    assert ev["career_snippets"][0]["title"] == "ML Engineer"


def test_extract_evidence_education_parsed():
    """extract_evidence parses education into structured format."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert len(ev["education"]) == 1
    assert ev["education"][0]["field_of_study"] == "computer science"
    assert ev["education"][0]["tier"] == "tier_1"


def test_extract_evidence_salary_parsed():
    """extract_evidence parses salary range from redrob_signals."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert ev["expected_salary_min"] == 25
    assert ev["expected_salary_max"] == 40


def test_extract_evidence_source_candidate_attached():
    """extract_evidence attaches the raw candidate as _source_candidate."""
    from src.evidence_builder import extract_evidence
    cand = _full_candidate()
    ev = extract_evidence(cand)
    assert "_source_candidate" in ev
    assert ev["_source_candidate"]["candidate_id"] == "CAND_001"


# === Retrieval ===

def test_dual_lane_returns_indices():
    """get_dual_lane_top_k returns a non-empty array of integer indices."""
    from src.retrieval import get_dual_lane_top_k
    e1 = _ev(id="A", penalties=[])
    e2 = _ev(id="B", penalties=[])
    e3 = _ev(id="C", penalties=[])
    fm = compile_feature_matrix([e1, e2, e3])
    penalties = np.zeros(3, dtype=np.float32)
    indices = get_dual_lane_top_k(fm, penalties)
    assert len(indices) > 0
    assert indices.dtype == np.int32


def test_dual_lane_no_duplicates():
    """get_dual_lane_top_k returns unique indices (union of two lanes)."""
    from src.retrieval import get_dual_lane_top_k
    candidates = [_ev(id=f"CAND_{i}", penalties=[]) for i in range(20)]
    fm = compile_feature_matrix(candidates)
    penalties = np.zeros(20, dtype=np.float32)
    indices = get_dual_lane_top_k(fm, penalties)
    assert len(indices) == len(set(indices))


def test_dual_lane_respects_max_limit():
    """get_dual_lane_top_k caps output at TOP_K_SEMANTIC."""
    from src.retrieval import get_dual_lane_top_k
    from src.config import CONFIG
    candidates = [_ev(id=f"CAND_{i}", penalties=[]) for i in range(5000)]
    fm = compile_feature_matrix(candidates)
    penalties = np.zeros(5000, dtype=np.float32)
    indices = get_dual_lane_top_k(fm, penalties)
    assert len(indices) <= CONFIG["TOP_K_SEMANTIC"]


def test_dual_lane_penalized_candidates_ranked_lower():
    """Candidates with high penalties should not appear in top indices."""
    from src.retrieval import get_dual_lane_top_k
    good = _ev(id="GOOD", penalties=[])
    bad = _ev(id="BAD", penalties=[])
    fm = compile_feature_matrix([good, bad])
    penalties = np.array([0.0, 0.45], dtype=np.float32)
    indices = get_dual_lane_top_k(fm, penalties)
    good_idx = 0
    bad_idx = 1
    if good_idx in indices and bad_idx in indices:
        assert list(indices).index(good_idx) < list(indices).index(bad_idx)


# === Streamer ===

def test_streamer_reads_jsonl(tmp_path):
    """stream_candidates yields dicts from a JSONL file."""
    from src.streamer import stream_candidates
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        '{"candidate_id": "C1", "profile": {}}\n'
        '{"candidate_id": "C2", "profile": {}}\n',
        encoding="utf-8"
    )
    results = list(stream_candidates(str(jsonl)))
    assert len(results) == 2
    assert results[0]["candidate_id"] == "C1"
    assert results[1]["candidate_id"] == "C2"


def test_streamer_skips_blank_lines(tmp_path):
    """stream_candidates skips empty lines gracefully."""
    from src.streamer import stream_candidates
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        '{"candidate_id": "C1"}\n'
        '\n'
        '   \n'
        '{"candidate_id": "C2"}\n',
        encoding="utf-8"
    )
    results = list(stream_candidates(str(jsonl)))
    assert len(results) == 2


def test_streamer_handles_malformed_json(tmp_path):
    """stream_candidates skips malformed JSON lines without crashing."""
    from src.streamer import stream_candidates
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        '{"candidate_id": "C1"}\n'
        'NOT VALID JSON\n'
        '{"candidate_id": "C2"}\n',
        encoding="utf-8"
    )
    results = list(stream_candidates(str(jsonl)))
    assert len(results) == 2


def test_streamer_max_candidates(tmp_path):
    """stream_candidates respects max_candidates limit."""
    from src.streamer import stream_candidates
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        '\n'.join(f'{{"candidate_id": "C{i}"}}' for i in range(10)),
        encoding="utf-8"
    )
    results = list(stream_candidates(str(jsonl), max_candidates=3))
    assert len(results) == 3


def test_streamer_gzip_support(tmp_path):
    """stream_candidates reads .gz files."""
    import gzip
    from src.streamer import stream_candidates
    gz_path = tmp_path / "test.jsonl.gz"
    content = '\n'.join(f'{{"candidate_id": "C{i}"}}' for i in range(5))
    with gzip.open(str(gz_path), 'wt', encoding='utf-8') as f:
        f.write(content)
    results = list(stream_candidates(str(gz_path)))
    assert len(results) == 5


# === CSV Writer ===

def test_csv_writer_creates_file(tmp_path):
    """write_submission creates a CSV file at the given path."""
    from src.csv_writer import write_submission
    out = tmp_path / "output" / "test.csv"
    results = [
        {"candidate_id": "CAND_001", "rank": 1, "score": 0.95, "reasoning": "Strong candidate."},
        {"candidate_id": "CAND_002", "rank": 2, "score": 0.90, "reasoning": "Good fit."},
    ]
    write_submission(results, str(out))
    assert out.exists()


def test_csv_writer_correct_content(tmp_path):
    """write_submission writes correct headers and rows."""
    from src.csv_writer import write_submission
    out = tmp_path / "test.csv"
    results = [
        {"candidate_id": "CAND_001", "rank": 1, "score": 0.95, "reasoning": "Top candidate."},
    ]
    write_submission(results, str(out))
    content = out.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert lines[0] == "candidate_id,rank,score,reasoning"
    assert "CAND_001" in lines[1]
    assert "1" in lines[1]


def test_csv_writer_empty_results(tmp_path):
    """write_submission handles empty results list."""
    from src.csv_writer import write_submission
    out = tmp_path / "test.csv"
    write_submission([], str(out))
    content = out.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 1  # Header only


def test_csv_writer_utf8_encoding(tmp_path):
    """write_submission uses UTF-8 encoding."""
    from src.csv_writer import write_submission
    out = tmp_path / "test.csv"
    results = [
        {"candidate_id": "CAND_001", "rank": 1, "score": 0.95, "reasoning": "Candidate with Unicode: \u00e9\u00e0\u00fc"},
    ]
    write_submission(results, str(out))
    content = out.read_text(encoding="utf-8")
    assert "\u00e9" in content


# === Profiling ===

def test_profiler_start_stop():
    """PipelineProfiler tracks phase timings."""
    from src.profiling import PipelineProfiler
    p = PipelineProfiler()
    p.start("Test Phase")
    p.stop()
    assert "Test Phase" in p.phases
    assert p.phases["Test Phase"] >= 0


def test_profiler_report():
    """PipelineProfiler.report() returns a string with phase info."""
    from src.profiling import PipelineProfiler
    p = PipelineProfiler()
    p.start("Phase A")
    p.stop()
    p.start("Phase B")
    p.stop()
    report = p.report()
    assert "Phase A" in report
    assert "Phase B" in report
    assert "TOTAL" in report


def test_profiler_auto_stops_previous():
    """PipelineProfiler.start() auto-stops the previous phase."""
    from src.profiling import PipelineProfiler
    p = PipelineProfiler()
    p.start("Phase A")
    p.start("Phase B")  # Should auto-stop Phase A
    assert "Phase A" in p.phases
    assert p._current_phase == "Phase B"


# === Integration Test ===

def _write_synthetic_jsonl(tmp_path, candidates):
    """Write a list of raw candidate dicts to a JSONL file."""
    import json
    path = tmp_path / "candidates.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    return str(path)


def _synthetic_candidate(cid, title="ML Engineer", yoe=7.0, skills=None,
                         career=None, signals=None, education=None):
    """Build a minimal but valid raw candidate dict for integration testing."""
    if skills is None:
        skills = [
            {"name": "faiss", "proficiency": "expert", "duration_months": 24},
            {"name": "python", "proficiency": "advanced", "duration_months": 36},
        ]
    if career is None:
        career = [
            {"company": "Flipkart", "title": title,
             "description": "built ranking with faiss and rag pipeline",
             "start_date": "2020-01-01", "duration_months": 60, "is_current": True}
        ]
    if signals is None:
        signals = {
            "recruiter_response_rate": 0.80,
            "github_activity_score": 75.0,
            "notice_period_days": 60,
            "open_to_work_flag": True,
            "saved_by_recruiters_30d": 5,
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
            "avg_response_time_hours": 12,
            "profile_completeness_score": 85,
            "interview_completion_rate": 0.70,
            "applications_submitted_30d": 3,
            "last_active_date": "2026-06-01",
        }
    if education is None:
        education = [
            {"field_of_study": "Computer Science", "tier": "tier_1", "degree": "b.tech"}
        ]
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "years_of_experience": yoe,
            "location": "Pune, Maharashtra",
            "country": "India",
            "headline": f"{title} at Flipkart",
        },
        "skills": skills,
        "career_history": career,
        "education": education,
        "redrob_signals": signals,
    }


def test_integration_evidence_to_csv(tmp_path):
    """Full integration: JSONL -> evidence -> features -> penalties -> retrieval -> aggregation -> CSV."""
    from src.evidence_builder import extract_evidence
    from src.feature_compiler import compile_feature_matrix
    from src.penalty_engine import apply_penalties
    from src.retrieval import get_dual_lane_top_k
    from src.score_aggregator import aggregate_scores
    from src.reason_generator import generate_reasoning
    from src.csv_writer import write_submission
    import json

    # 1. Create synthetic dataset
    candidates = [
        _synthetic_candidate("CAND_001", title="Senior ML Engineer", yoe=8.0),
        _synthetic_candidate("CAND_002", title="AI Researcher", yoe=5.0,
            career=[{"company": "Google", "title": "AI Researcher",
                     "description": "deep learning models", "start_date": "2021-01-01",
                     "duration_months": 48, "is_current": True}]),
        _synthetic_candidate("CAND_003", title="Data Scientist", yoe=3.0,
            signals={"recruiter_response_rate": 0.20, "github_activity_score": 10.0,
                     "notice_period_days": 90, "open_to_work_flag": False,
                     "saved_by_recruiters_30d": 0, "preferred_work_mode": "remote",
                     "willing_to_relocate": False, "expected_salary_range_inr_lpa": {"min": 10, "max": 15},
                     "avg_response_time_hours": 120, "profile_completeness_score": 30,
                     "interview_completion_rate": 0.10, "applications_submitted_30d": 0,
                     "last_active_date": "2025-01-01"}),
        _synthetic_candidate("CAND_004", title="ML Engineer", yoe=10.0,
            skills=[{"name": "pinecone", "proficiency": "expert", "duration_months": 36},
                    {"name": "rag", "proficiency": "advanced", "duration_months": 24}]),
        _synthetic_candidate("CAND_005", title="Consultant", yoe=6.0,
            career=[{"company": "TCS", "title": "Consultant",
                     "description": "client consulting work", "start_date": "2018-01-01",
                     "duration_months": 72, "is_current": True}],
            skills=[{"name": "java", "proficiency": "advanced", "duration_months": 60}]),
    ]

    jsonl_path = _write_synthetic_jsonl(tmp_path, candidates)

    # 2. Stream
    from src.streamer import stream_candidates
    raw = list(stream_candidates(jsonl_path))
    assert len(raw) == 5

    # 3. Evidence
    evidence_list = [extract_evidence(c) for c in raw]
    assert len(evidence_list) == 5
    assert evidence_list[0]["id"] == "CAND_001"

    # 4. Features
    feature_matrix = compile_feature_matrix(evidence_list)
    assert feature_matrix.shape == (5, 20)
    assert np.all(feature_matrix >= 0.0) and np.all(feature_matrix <= 1.0)

    # 5. Penalties
    penalties = apply_penalties(evidence_list)
    assert penalties.shape == (5,)
    # CAND_005 (TCS consultant, no production) should get hard filtered
    assert penalties[4] >= 0.50

    # 6. Retrieval
    top_indices = get_dual_lane_top_k(feature_matrix, penalties)
    assert len(top_indices) > 0
    assert len(top_indices) <= 5

    # 7. Semantic (skip — no model in test env)
    semantic_scores = np.zeros(len(top_indices), dtype=np.float32)

    # 8. Behavior
    from src.behavior_engine import compute_behavior_multiplier
    behavior_multipliers = compute_behavior_multiplier(evidence_list)

    # 9. Aggregation
    base_scores = (feature_matrix @ np.array(WEIGHTS, dtype=np.float32)) - penalties
    final_top_k = aggregate_scores(
        base_scores, semantic_scores, behavior_multipliers, penalties[top_indices],
        evidence_list, top_indices, target_k=5
    )
    assert len(final_top_k) > 0
    assert len(final_top_k) <= 5

    # 10. Reasoning
    for cand in final_top_k:
        idx = cand["_idx"]
        cand["reasoning"] = generate_reasoning(evidence_list[idx], rank=cand["rank"])
        assert len(cand["reasoning"]) > 0

    # 11. CSV output
    out_path = tmp_path / "output.csv"
    write_submission(final_top_k, str(out_path))
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert lines[0] == "candidate_id,rank,score,reasoning"
    assert len(lines) > 1  # At least header + 1 data row

    # 12. Verify rank order
    for line in lines[1:]:
        parts = line.split(",")
        rank = int(parts[1])
        assert 1 <= rank <= 5


def test_integration_full_pipeline_from_jsonl(tmp_path):
    """End-to-end: write JSONL -> stream -> full pipeline -> validate CSV output structure."""
    from src.streamer import stream_candidates
    from src.evidence_builder import extract_evidence
    from src.feature_compiler import compile_feature_matrix
    from src.penalty_engine import apply_penalties
    from src.retrieval import get_dual_lane_top_k
    from src.behavior_engine import compute_behavior_multiplier
    from src.score_aggregator import aggregate_scores
    from src.csv_writer import write_submission
    from src.config import CONFIG

    # Create 10 candidates with varying quality
    candidates = []
    for i in range(10):
        cid = f"CAND_{i:03d}"
        title = ["ML Engineer", "Data Scientist", "AI Researcher", "Search Engineer",
                 "Consultant"][i % 5]
        yoe = 2.0 + i * 1.5
        signals = {
            "recruiter_response_rate": min(0.3 + i * 0.07, 1.0),
            "github_activity_score": float(i * 10),
            "notice_period_days": 30 + i * 10,
            "open_to_work_flag": i % 2 == 0,
            "saved_by_recruiters_30d": i,
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "expected_salary_range_inr_lpa": {"min": 15, "max": 35},
            "avg_response_time_hours": max(24, 100 - i * 10),
            "profile_completeness_score": 50 + i * 5,
            "interview_completion_rate": min(0.3 + i * 0.07, 1.0),
            "applications_submitted_30d": i % 4,
            "last_active_date": f"2026-{6 - (i % 6):02d}-01",
        }
        candidates.append(_synthetic_candidate(cid, title=title, yoe=yoe, signals=signals))

    jsonl_path = _write_synthetic_jsonl(tmp_path, candidates)

    # Stream
    raw = list(stream_candidates(jsonl_path))
    assert len(raw) == 10

    # Extract
    evidence_list = [extract_evidence(c) for c in raw]

    # Features
    feature_matrix = compile_feature_matrix(evidence_list)
    assert feature_matrix.shape == (10, 20)

    # Penalties
    penalties = apply_penalties(evidence_list)

    # Retrieval
    top_indices = get_dual_lane_top_k(feature_matrix, penalties)
    assert len(top_indices) <= CONFIG["TOP_K_SEMANTIC"]

    # Behavior
    behavior_multipliers = compute_behavior_multiplier(evidence_list)

    # Aggregate
    semantic_scores = np.zeros(len(top_indices), dtype=np.float32)
    base_scores = (feature_matrix @ np.array(WEIGHTS, dtype=np.float32)) - penalties
    final_top_k = aggregate_scores(
        base_scores, semantic_scores, behavior_multipliers, penalties[top_indices],
        evidence_list, top_indices, target_k=100
    )

    # Output
    out_path = tmp_path / "output.csv"
    write_submission(final_top_k, str(out_path))
    content = out_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert lines[0] == "candidate_id,rank,score,reasoning"
    # Should have at most 10 data rows
    assert len(lines) - 1 <= 10
    assert len(lines) - 1 > 0


# === enforce_monotonicity ===

def test_enforce_monotonicity_already_sorted():
    """_enforce_monotonicity should not modify an already-sorted list."""
    from src.main import _enforce_monotonicity
    items = [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}]
    _enforce_monotonicity(items)
    assert [i["score"] for i in items] == [0.9, 0.8, 0.7]

def test_enforce_monotonicity_clamps_violation():
    """_enforce_monotonicity clamps a score that violates monotonicity."""
    from src.main import _enforce_monotonicity
    items = [{"score": 0.9}, {"score": 0.95}, {"score": 0.7}]
    _enforce_monotonicity(items)
    assert items[1]["score"] == 0.9  # Clamped to previous
    assert items[2]["score"] == 0.7

def test_enforce_monotonicity_empty():
    """_enforce_monotonicity handles empty list."""
    from src.main import _enforce_monotonicity
    items = []
    _enforce_monotonicity(items)
    assert items == []

def test_enforce_monotonicity_single():
    """_enforce_monotonicity handles single element."""
    from src.main import _enforce_monotonicity
    items = [{"score": 0.5}]
    _enforce_monotonicity(items)
    assert items[0]["score"] == 0.5


# === Aliases ===

def test_aliases_all_keys_compile():
    """All keys in COMPILED_ALIASES should be compiled regex patterns."""
    import re
    from src.aliases import COMPILED_ALIASES
    for key, pattern in COMPILED_ALIASES.items():
        assert isinstance(pattern, re.Pattern), f"Key '{key}' is not a compiled regex"

def test_aliases_rag_matches():
    """COMPILED_ALIASES['rag'] should match 'rag' and 'RAG'."""
    from src.aliases import COMPILED_ALIASES
    pattern = COMPILED_ALIASES["rag"]
    assert pattern.search("rag")
    assert pattern.search("RAG")
    assert pattern.search("built a rag pipeline")

def test_aliases_no_false_positive():
    """Aliases should not match unrelated substrings."""
    from src.aliases import COMPILED_ALIASES
    pattern = COMPILED_ALIASES["rag"]
    assert not pattern.search("stage")
