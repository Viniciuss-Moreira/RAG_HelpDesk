import numpy as np
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import HybridReranker

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

def main():
    # Carrega os textos (chunks)
    with open("data/raw/base_treinamento.txt", encoding="utf-8") as f:
        chunks = [b.strip() for b in f.read().split("\n\n") if b.strip()]

    # Cria o retriever com os embeddings pré-calculados
    retriever = Retriever("data/embeddings/batch_000.npy")

    # Cria o reranker, passando o retriever e os chunks para reranking
    hybrid = HybridReranker(
        retriever=retriever,
        chunk_texts=chunks,
        reranker_model=RERANKER_MODEL,
        sparse_alpha=0.5,
    )

    query = "Meu notebook está muito lento, o que posso fazer? E se não resolver?"

    # Recupera e reranqueia os top documentos
    idxs, scores = hybrid.retrieve_and_rerank(query, top_k_dense=10, top_k_final=3)

    print(f"\nQuery: {query}\n")
    for i, idx in enumerate(idxs):
        print(f"Rank {i+1} - Chunk #{idx} (score: {scores[i]:.4f}):")
        print(chunks[idx])
        print("-" * 40)

if __name__ == "__main__":
    main()
