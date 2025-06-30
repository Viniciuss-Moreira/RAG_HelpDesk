import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

EMBEDDING_DIR = "data/embeddings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_embedder(model_name: str = MODEL_NAME):
    return SentenceTransformer(model_name)

def generate_embeddings(text_chunks: List[str], model) -> np.ndarray:
    embeddings = model.encode(text_chunks, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)

def save_embeddings(embeddings: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, embeddings)

def get_chunk_texts(chunks: List[dict]) -> List[str]:
    return [chunk["content"] for chunk in chunks]