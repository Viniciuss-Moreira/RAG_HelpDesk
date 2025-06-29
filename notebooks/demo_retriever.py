import numpy as np
from src.retrieval.retriever import Retriever

retriever = Retriever("data/embeddings/batch_000.npy")

query = "O que fazer se o notebook não liga?"
idxs, scores = retriever.retrieve(query, top_k=5)

idxs = np.atleast_1d(idxs).flatten()
scores = np.atleast_1d(scores).flatten()

print(f"\ntop results for: {query}\n")
for i, (idx, score) in enumerate(zip(idxs, scores), 1):
    print(f"{i}. idx: {int(idx):4d} — similarity: {score:.4f}")
