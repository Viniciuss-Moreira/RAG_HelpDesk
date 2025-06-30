import pytest
from src.retrieval.reranker import HybridReranker
from src.retrieval.retriever import Retriever

@pytest.fixture
def retriever():
    retriever = Retriever("data/embeddings/batch_000.npy")
    retriever.set_chunk_texts([
        "Texto do chunk 1",
        "Texto do chunk 2",
        "Texto do chunk 3",
        # Ajuste para ter o mesmo número de chunks do embeddings
    ])
    return retriever

@pytest.fixture
def reranker(retriever):
    chunk_texts = retriever.get_chunk_texts()
    return HybridReranker(retriever=retriever, chunk_texts=chunk_texts)

def test_reranker_retrieve_and_rerank(reranker):
    query = "Como resolver problema X?"
    idxs, scores = reranker.retrieve_and_rerank(query, top_k_dense=2, top_k_final=2)
    assert len(idxs) <= 2
    assert len(scores) == len(idxs)

def test_reranker_ranking_changes_with_query(reranker):
    query1 = "Problema com Wi-Fi"
    query2 = "Erro na instalação"
    idxs1, _ = reranker.retrieve_and_rerank(query1, top_k_dense=3, top_k_final=3)
    idxs2, _ = reranker.retrieve_and_rerank(query2, top_k_dense=3, top_k_final=3)
    assert idxs1 != idxs2
