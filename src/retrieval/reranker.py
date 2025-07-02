from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class HybridReranker:
    def __init__(
        self,
        vector_store: FAISS,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = 'cpu',
        cache_dir: str = "/app/huggingface_cache"
    ):
        self.vector_store = vector_store
        
        print(f"loading CrossEncoder. saving in: {cache_dir}")
        self.reranker = CrossEncoder(reranker_model, max_length=512, device=device, cache_folder=cache_dir)
        
        docs_in_order = list(self.vector_store.docstore._dict.values())
        self.chunk_texts = [doc.page_content for doc in docs_in_order]
        self.chunk_metadata = [doc.metadata for doc in docs_in_order]
        
        print("building tf-idf")
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        print("reranker ready")

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_final: int = 5,
    ) -> List[Document]:
        dense_docs = self.vector_store.similarity_search(query, k=top_k_dense)
        q_vec = self.vectorizer.transform([query])
        sparse_scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        sparse_indices = np.argsort(-sparse_scores)[:top_k_dense]
        sparse_docs = [
            Document(page_content=self.chunk_texts[i], metadata=self.chunk_metadata[i]) 
            for i in sparse_indices
        ]
        combined_docs = []
        seen_contents = set()
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)
        pairs = [[query, doc.page_content] for doc in combined_docs]
        rerank_scores = self.reranker.predict(pairs)
        doc_scores = list(zip(combined_docs, rerank_scores))
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in sorted_doc_scores[:top_k_final]]
        return final_docs