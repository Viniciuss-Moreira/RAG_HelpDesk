from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text
from src.ingestion.embedder import load_embedder, generate_embeddings, save_embeddings, get_chunk_texts

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RAW_PATH = "data/raw"
EMBEDDING_PATH = "data/embeddings/batch_000.npy"

docs = load_documents_from_dir(RAW_PATH)
clean_docs = preprocess_documents(docs)
chunks = split_text(clean_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

texts = get_chunk_texts(chunks)

model = load_embedder()
embeddings = generate_embeddings(texts, model)

print(f"embeddings shape: {embeddings.shape}")
print(f"vector example (1º):\n{embeddings[0][:10]}...")

save_embeddings(embeddings, EMBEDDING_PATH)
print(f"save embeddings in: {EMBEDDING_PATH}")