import numpy as np
from sentence_transformers import SentenceTransformer
from src.retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self, embeddings_path: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        self.index = VectorStore(dim=dim)

        embeds = np.load(embeddings_path).astype(np.float32)
        self.index.add(embeds)

    def retrieve(self, query: str, top_k: int = 5):
        qv = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
        distances, indices = self.index.search(qv, top_k)
        return indices[0].copy(), distances[0].copy()
    
    def set_chunk_texts(self, texts: list[str]):
        self._chunk_texts = texts

    def get_chunk_texts(self) -> list[str]:
        return getattr(self, "_chunk_texts", [])