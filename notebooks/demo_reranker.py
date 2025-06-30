import numpy as np
from sentence_transformers import CrossEncoder
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import HybridReranker
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text

RAW_PATH = "data/raw"
EMBEDDINGS_PATH = "data/embeddings/batch_000.npy"
MODEL_NAME     = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

docs = load_documents_from_dir(RAW_PATH)
clean_docs = preprocess_documents(docs)
chunks = split_text(clean_docs, chunk_size=300, chunk_overlap=50)
chunk_texts = [chunk['content'] for chunk in chunks]
print(f"total chunks: {len(chunk_texts)}\n")

retriever = Retriever(EMBEDDINGS_PATH, model_name=MODEL_NAME)
retriever.set_chunk_texts(chunk_texts)

hybrid = HybridReranker(retriever, reranker_model=RERANKER_MODEL, sparse_alpha=0.5)

query = "Como instalar uma impressora no Windows?"
print(f"query: {query}\n")

final_idxs, final_scores = hybrid.retrieve_and_rerank(
    query=query,
    top_k_dense=20,
    top_k_final=5,
    keyword_filter=["impressora", "windows"]
)

print("results after hybrid rerank:\n")
for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), 1):
    print(f" {rank}. idx: {idx:5d} — score: {score:.4f}")
    print(f"    {chunk_texts[idx][:200]}...\n")