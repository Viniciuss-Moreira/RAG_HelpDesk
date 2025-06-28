import numpy as np
from src.retrieval.vector_store import VectorStore

embeds = np.load("data/embeddings/batch_000.npy")
dim = embeds.shape[1]
print(f"embeddings loaded: {embeds.shape}")

vs = VectorStore(dim=dim)
vs.add(embeds)
print("vectors added in index")

query = embeds[0]
dists, idxs = vs.search(query, top_k=5)

print("\nresult search (dummy):")
for i, (idx, dist) in enumerate(zip(idxs, dists), 1):
    print(f"{i}. idx: {idx:4d} — distância: {dist:.4f}")

print("Norma do primeiro vetor:", np.linalg.norm(embeds[0]))
print("Norma do segundo vetor:", np.linalg.norm(embeds[1]))
