from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class HybridReranker:
    def __init__(
        self,
        vector_store: FAISS,
        # ####################################################################
        # ## CORREÇÃO FINAL: NOME DO MODELO CORRETO E VERIFICADO          ##
        # ####################################################################
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Inicializa com o Vector Store do LangChain e um modelo de re-ranking multilíngue.
        """
        self.vector_store = vector_store
        self.reranker = CrossEncoder(reranker_model)
        
        self.chunk_texts = [
            doc.page_content for doc in self.vector_store.docstore._dict.values()
        ]
        
        print(f"Modelo de Re-ranking '{reranker_model}' carregado. Construindo matriz TF-IDF...")
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        print("HybridReranker pronto.")

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_final: int = 5,
    ) -> List[Document]:
        # A lógica deste método não precisa de alterações
        dense_docs = self.vector_store.similarity_search(query, k=top_k_dense)
        
        q_vec = self.vectorizer.transform([query])
        sparse_scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        sparse_indices = np.argsort(-sparse_scores)[:top_k_dense]
        sparse_docs = [
            Document(page_content=self.chunk_texts[i]) for i in sparse_indices
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