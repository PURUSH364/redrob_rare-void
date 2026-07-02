# jd_matrix.py — Single source of truth for JD requirements.
# Never duplicate keyword lists elsewhere.

JD_MATRIX = {
    "critical": {
        "vector_db": [
            "pinecone", "milvus", "qdrant", "weaviate", "chroma", "faiss", "vespa"
        ],
        "retrieval": [
            "rag", "semantic search", "retrieval augmented generation", "vector search", "dense retrieval"
        ],
        "evaluation": [
            "ragas", "truera", "arize", "mlflow", "ndcg", "mrr", "offline evaluation"
        ],
        "python": [
            "python", "fastapi", "flask", "django"
        ]
    },
    "strong": [
        "recsys", "recommendation system", "learning to rank", "ltr", "fine-tuning", "search ranking", "matching system",
        "peft", "lora", "qlora", "shipped", "deployed", "production", "at scale", "latency", "a/b testing"
    ],
    "negative": [
        "langchain_only", "research_only", "consulting_only"
    ]
}
