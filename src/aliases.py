import re

# aliases.py — Explicit query expansion and compiled regexes for aliases
# Improves recall and handles multi-word aliases correctly.

ALIASES = {
    "rag": [
        r"retrieval augmented generation",
        r"retrieval-augmented generation",
        r"\brag\b"
    ],
    "semantic_search": [
        r"semantic search",
        r"vector search",
        r"dense retrieval"
    ],
    "offline_evaluation": [
        r"offline evaluation",
        r"offline metrics",
        r"ndcg",
        r"mrr"
    ],
    "ab_testing": [
        r"a/b testing",
        r"ab testing",
        r"online evaluation"
    ],
    "production_evidence": [
        r"\bdeployed\b",
        r"\bshipped\b",
        r"\bproduction\b",
        r"at scale",
        r"latency",
        r"sla"
    ],
    "recsys": [
        r"\brecsys\b",
        r"recommendation system",
        r"recommendation engine",
        r"recommender system",
        r"collaborative filtering"
    ],
    "ltr": [
        r"\bltr\b",
        r"learning to rank",
        r"learning-to-rank",
        r"learn to rank"
    ],
    "fine_tuning": [
        r"fine-tuning",
        r"fine tuning",
        r"\bfinetuning\b",
        r"\bfine-tune\b",
        r"\bfinetune\b",
        r"\bpeft\b",
        r"\blora\b",
        r"\bqlora\b"
    ]
}

# Compile aliases to regexes
COMPILED_ALIASES = {
    key: re.compile('|'.join(patterns), re.IGNORECASE)
    for key, patterns in ALIASES.items()
}
