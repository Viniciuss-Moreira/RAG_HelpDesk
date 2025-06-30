# scripts/gerar_benchmark.py
import json
from sentence_transformers import SentenceTransformer, util
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text

# 1. Define queries
queries = [
    "Como liberar espaço em disco?",
    "O que é firewall e como ativar?",
    "Como formatar meu notebook com Windows 10?",
    "Como conectar impressora à rede Wi-Fi?",
    "Por que o Wi-Fi desconecta sozinho?"
]

# 2. Carrega e processa documentos
print("🔄 Carregando e processando documentos...")
docs = load_documents_from_dir("data/raw")
clean_docs = preprocess_documents(docs)
chunks = split_text(clean_docs, chunk_size=300, chunk_overlap=50)
texts = [chunk['content'] for chunk in chunks]

# 3. Embeddings com SentenceTransformer
print("🔎 Gerando embeddings com SentenceTransformer...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
chunk_embeddings = model.encode(texts, convert_to_tensor=True)

# 4. Encontra o chunk mais relevante para cada query
print("📌 Gerando benchmark...")
benchmark = []
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    best_idx = int(scores.argmax())
    benchmark.append({
        "query": query,
        "relevant_idxs": [best_idx]
    })

# 5. Salva benchmark
with open("tests/benchmark.json", "w", encoding="utf-8") as f:
    json.dump(benchmark, f, ensure_ascii=False, indent=2)

print("✅ Benchmark salvo em tests/benchmark.json")
