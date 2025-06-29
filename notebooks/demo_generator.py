import numpy as np
from src.retrieval.retriever import Retriever
from src.generation.response_generator import generate_answer

retriever = Retriever("data/embeddings/batch_000.npy")
query = "Meu notebook está muito lento, o que posso fazer? E se não resolver?"

idxs, scores = retriever.retrieve(query, top_k=1)

idxs = np.atleast_1d(idxs).flatten()
scores = np.atleast_1d(scores).flatten()

with open("data/raw/base_treinamento.txt", encoding="utf-8") as f:
    chunks = [b.strip() for b in f.read().split("\n\n") if b.strip()]

context = chunks[int(idxs[0])]

answer = generate_answer(query, context)

print(f"\nquery: {query}")
print(f"context selected (chunk #{idxs[0]}):\n{context}\n")
print("generated response:\n")
print(answer)
