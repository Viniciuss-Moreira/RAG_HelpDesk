import numpy as np
from sentence_transformers import SentenceTransformer
from src.retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self, embeddings_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        self.index = VectorStore(dim=dim)

        embeds = np.load(embeddings_path).astype(np.float32)
        self.index.add(embeds)

    def retrieve(self, query: str, top_k: int = 5):
        qv = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
        distances, indices = self.index.search(qv, top_k)
        return indices[0], distances[0]
