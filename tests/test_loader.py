from src.ingestion.document_loader import load_documents_from_dir

docs = load_documents_from_dir("data/raw")

print(f"docs loaded: {len(docs)}")

for doc in docs:
    print(f"\nfont: {doc['source']}")
    print(f"first letters:\n{doc['content'][:200]}")