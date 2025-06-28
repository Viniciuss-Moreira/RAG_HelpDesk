from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text

RAW_PATH = "data/raw"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

docs = load_documents_from_dir(RAW_PATH)
print(f"docs loaded: {len(docs)}")

cleaned_docs = preprocess_documents(docs)
print(f"pre-process completed")

chunks = split_text(cleaned_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
print(f"total chunks generated: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ---")
    print(f"font: {chunk['source']}")
    print(chunk['content'][:300])