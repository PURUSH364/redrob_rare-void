# config.py — Global configuration parameters

from datetime import datetime

REFERENCE_DATE = datetime(2026, 6, 21)

CONFIG = {
    # Funnel bounds
    "TOP_K_AI": 2000,
    "TOP_K_PRODUCTION": 2000,
    "TOP_K_SEMANTIC": 4000,
    "TOP_K_FINAL": 100,

    # Weights
    "FEATURE_WEIGHT": 0.60,
    "SEMANTIC_WEIGHT": 0.40,

    # Behavior Multiplier bounds (~1.64:1 ratio so strong behavioral
    # signals can meaningfully reward/penalize candidates)
    "BEHAVIOR_MIN": 0.70,
    "BEHAVIOR_MAX": 1.15,

    # Maximum soft penalty (0-1 scale, magnitude — always positive)
    "MAX_PENALTY": 0.50
}

PREFERRED_CITIES_TIER1 = {'pune', 'noida'}
PREFERRED_CITIES_TIER2 = {'hyderabad', 'mumbai', 'delhi', 'gurugram', 'bangalore', 'bengaluru'}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "hcl", "cognizant",
    "accenture", "capgemini", "ibm", "tech mahindra", "deloitte", "pwc",
    "ernst & young", "ey", "kpmg", "mckinsey", "bcg", "bain"
}

LD_KEYWORDS = {'langchain', 'llamaindex', 'openai api', 'chatgpt api'}

# Behavior engine sub-weights (must sum to 1.0)
BEHAVIOR_WEIGHTS = {
    "W_RESPONSE": 0.35,
    "W_GITHUB": 0.20,
    "W_AVAILABILITY": 0.25,
    "W_DEMAND": 0.20,
}

# Pre-LLM ML experience detection pattern
import re
PRE_LLM_PATTERN = re.compile(r'\b(?:machine learning|ml|ai|nlp|model)\b', re.IGNORECASE)

# Pre-LLM cutoff date for experience detection
PRE_LLM_CUTOFF_DATE = "2022-01-01"

# YoE fit Gaussian parameters (center, sigma)
YOE_FIT_CENTER = 7.0
YOE_FIT_SIGMA = 2.5

# Recency decay parameters
RECENCY_MAX_DAYS = 180.0
RECENCY_DECAY_FACTOR = 0.8
RECENCY_FLOOR = 0.2

# Salary range for fit scoring (LPA)
SALARY_RANGE_MIN = 15
SALARY_RANGE_MAX = 50

# Title classification for feature scoring
GOOD_TITLES = ["machine learning", "ai", "data scientist", "ml engineer",
               "nlp engineer", "search engineer", "ranking engineer",
               "recommendation engineer"]
BAD_TITLES = ["marketing manager", "content writer", "mechanical engineer",
              "civil engineer", "accountant"]

# Relevant industries for feature scoring
RELEVANT_INDUSTRIES = ["e-commerce", "ecommerce", "fintech", "saas",
    "search", "ads", "advertising", "marketplace", "social media",
    "streaming", "payments", "logistics", "healthtech"]

# The feature weights for the vectorised compilation phase
# These sum to 1.0 to ensure the base feature score is naturally bounded [0, 1]
WEIGHTS = [
    # Critical (4 features, ~50%)
    0.15,  # F0: vector_db
    0.15,  # F1: retrieval_rag
    0.10,  # F2: eval
    0.10,  # F3: python

    # Strong (3 features, ~17%)
    0.06,  # F4: recsys
    0.06,  # F5: ltr
    0.05,  # F6: fine_tuning

    # Career & Logistics (8 features, ~23%)
    0.05,  # F7: prod_depth
    0.05,  # F8: product_co_ratio
    0.04,  # F9: yoe_fit
    0.04,  # F10: title_align
    0.01,  # F11: recency
    0.01,  # F12: tenure
    0.01,  # F13: shipped_sys
    0.02,  # F14: industry_fit

    # New additions (5 features, ~10%)
    0.02,  # F15: location_fit
    0.03,  # F16: education_fit
    0.02,  # F17: work_mode_fit
    0.01,  # F18: salary_fit
    0.02,  # F19: pre_llm_ml_experience
]

NUM_FEATURES = 20  # Total features (0-19) used in dot product
assert len(WEIGHTS) == NUM_FEATURES, f"WEIGHTS length ({len(WEIGHTS)}) must match NUM_FEATURES ({NUM_FEATURES})"
