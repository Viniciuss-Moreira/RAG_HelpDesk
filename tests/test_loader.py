from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.text_splitter import split_text

docs = load_documents_from_dir("data/raw")

print(f"docs loaded: {len(docs)}")

for doc in docs:
    print(f"\nfont: {doc['source']}")
    print(f"first letters:\n{doc['content'][:200]}")

docs = load_documents_from_dir("data/raw")
chunks = split_text(docs, chunk_size=300, chunk_overlap=50)

print(f"total chunks generated: {len(chunks)}")
print(f"example:\n{chunks[0]}")
