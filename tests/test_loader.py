from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.text_splitter import split_text

docs = load_documents_from_dir("data/raw")
cleaned_docs = preprocess_documents(docs)
chunks = split_text(cleaned_docs)

print(chunks[0]["content"])
