import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)
        return distances[0], indices[0]