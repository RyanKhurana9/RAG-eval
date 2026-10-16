import json
import random

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import OllamaModel
from langchain_ollama import OllamaEmbeddings

from src.retriever import build_retriever


# -----------------------------
# 1. CONFIG
# -----------------------------

GOLDEN_PATH = "goldens/retriever_goldens.json"
THRESHOLD = 0.7

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

judge_model = OllamaModel(
    model="llama3.1:8b"
)


# -----------------------------
# 2. LOAD GOLDENS
# -----------------------------

with open(GOLDEN_PATH, "r") as f:
    goldens = json.load(f)


# -----------------------------
# 3. BUILD RETRIEVER
# -----------------------------

retriever = build_retriever()

test_cases = []

for g in random.sample(goldens,4):

    retrieved = retriever.invoke(g["query"])

    retrieval_context = [
        doc.page_content
        for doc in retrieved
    ]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            actual_output="(generator not evaluated in this run)",
            retrieval_context=retrieval_context,
        )
    )


# -----------------------------
# 4. METRICS
# -----------------------------

metrics = [
    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=True,
    ),

    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=True,
    ),
]


# -----------------------------
# 5. EVALUATE
# -----------------------------

result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "reranker",
        "embedding_model": "nomic-embed-text",
        "chunk_size": 1000,
        "top_k": 3,
        "judge_model": "llama3.1:8b",
        "golden_set": GOLDEN_PATH,
    },
)