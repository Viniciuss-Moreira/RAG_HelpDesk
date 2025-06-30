import json
import numpy as np
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import HybridReranker
from src.evaluation.metrics import (
    precision_at_k as retrieval_precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    bleu_score
)
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text


def run_evaluation(
    benchmark_path: str = "tests/benchmark.json",
    k: int = 3,
    top_k_dense: int = 10,
    top_k_final: int = 3,
    sparse_alpha: float = 0.5
):
    with open(benchmark_path, encoding="utf-8") as f:
        benchmarks = json.load(f)

    docs = load_documents_from_dir("data/raw")
    clean_docs = preprocess_documents(docs)
    chunks = split_text(clean_docs, chunk_size=300, chunk_overlap=50)
    texts = [chunk['content'] for chunk in chunks]

    retriever = Retriever("data/embeddings/batch_000.npy")
    reranker = HybridReranker(
        retriever=retriever,
        chunk_texts=texts,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        sparse_alpha=sparse_alpha
    )

    all_retrieved = []
    all_relevant = []

    print(f"starting assessment reranker: Precision@{k}, Recall@{k}, MRR")
    print(f"dense top_k: {top_k_dense}, final top_k: {top_k_final}, sparse_alpha: {sparse_alpha}\n")

    for i, entry in enumerate(benchmarks, 1):
        query = entry['query']
        relevant_idxs = entry.get('relevant_idxs', [])

        idxs, scores = reranker.retrieve_and_rerank(
            query,
            top_k_dense=top_k_dense,
            top_k_final=top_k_final
        )

        p = retrieval_precision_at_k(retrieved_idxs=idxs, relevant_idxs=relevant_idxs, k=k)
        r = recall_at_k(retrieved_idxs=idxs, relevant_idxs=relevant_idxs, k=k)

        all_retrieved.append(idxs)
        all_relevant.append(relevant_idxs)

        print(f"{i}. Query: {query}")
        print(f"   Precision@{k}: {p:.2f}, Recall@{k}: {r:.2f}")
        print(f"   Retrieved idxs: {idxs}")
        print(f"   Rerank scores: {[f'{s:.4f}' for s in scores]}\n")

    mrr = mean_reciprocal_rank(retrieved_lists=all_retrieved, relevant_idxs_list=all_relevant)
    print(f"mean reciprocal rank (MRR): {mrr:.2f}\n")

if __name__ == "__main__":
    run_evaluation()