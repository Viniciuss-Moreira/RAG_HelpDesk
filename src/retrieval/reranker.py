from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from src.retrieval.retriever import Retriever

class HybridReranker:
    def __init__(
        self,
        retriever: Retriever,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        sparse_alpha: float = 0.5
    ):

        self.retriever = retriever
        self.reranker = CrossEncoder(reranker_model)
        self.sparse_alpha = sparse_alpha

        chunk_texts = self.retriever.get_chunk_texts()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_final: int = 5,
        keyword_filter: Optional[List[str]] = None
    ) -> Tuple[List[int], List[float]]:

        dense_idxs, dense_scores = self.retriever.retrieve(query, top_k=top_k_dense)
        dense_idxs = np.atleast_1d(dense_idxs).flatten().tolist()

        if keyword_filter:
            dense_idxs = [i for i in dense_idxs
                          if all(kw.lower() in self.retriever.get_chunk_texts()[i].lower()
                                 for kw in keyword_filter)]

        q_vec = self.vectorizer.transform([query])
        sparse_scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        sparse_idxs = np.argsort(-sparse_scores)[:top_k_dense].tolist()

        combined = []
        for idx in dense_idxs + sparse_idxs:
            if idx not in combined:
                combined.append(idx)
            if len(combined) >= top_k_dense:
                break

        candidate_texts = [self.retriever.get_chunk_texts()[i] for i in combined]
        pairs = [(query, chunk.split("Resposta:")[0].strip()) for chunk in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)
        order = np.argsort(rerank_scores)[::-1]

        final_idxs = [combined[i] for i in order[:top_k_final]]
        final_scores = [float(rerank_scores[i]) for i in order[:top_k_final]]
        return final_idxs, final_scores
