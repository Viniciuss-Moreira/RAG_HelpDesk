import numpy as np
from src.retrieval.retriever import Retriever

def test_retrieve_top_k():
    retriever = Retriever("data/embeddings/batch_000.npy")
    idxs, scores = retriever.retrieve("test query", top_k=3)

    idxs = np.atleast_1d(idxs)
    scores = np.atleast_1d(scores)

    assert len(idxs) > 0
    assert len(scores) == len(idxs)
