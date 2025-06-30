import json
from pathlib import Path
import numpy as np
from src.retrieval.retriever import Retriever
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text


def precision_at_k(expected_keywords, retrieved_texts, k=3):
    hits = 0
    top_texts = retrieved_texts[:k]
    joined = " ".join(top_texts).lower()
    for kw in expected_keywords:
        if kw.lower() in joined:
            hits += 1
    return hits / len(expected_keywords) if expected_keywords else 0


def run_evaluation(
    benchmark_path: str = "tests/benchmark.json",
    k: int = 3
):
    # 1. Carrega benchmarks
    with open(benchmark_path, encoding="utf-8") as f:
        benchmarks = json.load(f)

    docs = load_documents_from_dir("data/raw")
    clean_docs = preprocess_documents(docs)
    chunks = split_text(clean_docs, chunk_size=300, chunk_overlap=50)
    texts = [chunk['content'] for chunk in chunks]

    retriever = Retriever("data/embeddings/batch_000.npy")

    precisions = []

    print(f"starting assessments with Precision@{k}\n")
    for i, entry in enumerate(benchmarks, 1):
        query = entry['query']
        expected = entry['expected_keywords']
        idxs, _ = retriever.retrieve(query, top_k=k)
        idxs = np.atleast_1d(idxs).flatten()

        retrieved_texts = []
        for j in idxs:
            j_int = int(j)
            if 0 <= j_int < len(texts):
                retrieved_texts.append(texts[j_int])
            else:
                print(f"[evaluator] warning: index {j_int} is out of range and will be ignored.")

        score = precision_at_k(expected, retrieved_texts, k)
        precisions.append(score)
        print(f"{i}. Query: {query}\n   Precision@{k}: {score:.2f}\n")

    avg = sum(precisions) / len(precisions) if precisions else 0
    print(f"media Precision@{k}: {avg:.2f}")