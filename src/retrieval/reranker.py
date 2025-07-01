from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from src.retrieval.retriever import Retriever

class HybridReranker:
    def __init__(
        self,
        retriever: Retriever,
        chunk_texts: List[str],
        reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384",
        sparse_alpha: float = 0.5
    ):
        self.retriever = retriever
        self.chunk_texts = chunk_texts
        self.n_chunks = len(chunk_texts)
        self.reranker = CrossEncoder(reranker_model)
        self.sparse_alpha = sparse_alpha

        try:
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            self.use_sparse = True
        except Exception:
            self.vectorizer = None
            self.tfidf_matrix = np.zeros((self.n_chunks, 1))
            self.use_sparse = False

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_final: int = 5,
        keyword_filter: Optional[List[str]] = None
    ) -> Tuple[List[int], List[float]]:
        dense_idxs, _ = self.retriever.retrieve(query, top_k=top_k_dense)
        dense_idxs = np.atleast_1d(dense_idxs).flatten().tolist()

        if keyword_filter:
            dense_idxs = [
                i for i in dense_idxs
                if all(kw.lower() in self.chunk_texts[i].lower() for kw in keyword_filter)
            ]

        if self.use_sparse:
            q_vec = self.vectorizer.transform([query])
            sparse_scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        else:
            sparse_scores = np.zeros(self.n_chunks)
        sparse_idxs = np.argsort(-sparse_scores)[:top_k_dense].tolist()

        combined = []
        for idx in dense_idxs + sparse_idxs:
            if idx not in combined and idx < self.n_chunks:  # Proteção contra índice inválido
                combined.append(idx)
            if len(combined) >= top_k_dense:
                break

        candidate_texts = [self.chunk_texts[i] for i in combined]
        pairs = [(query, c.split("Resposta:")[0].strip()) for c in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)
        order = np.argsort(rerank_scores)[::-1]

        final_idxs = [combined[i] for i in order[:top_k_final]]
        final_scores = [float(rerank_scores[i]) for i in order[:top_k_final]]
        return final_idxs, final_scores
