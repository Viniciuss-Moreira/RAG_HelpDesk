import os
import shutil
import torch
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def check_environment():

    if not torch.cuda.is_available():
        print("\nCUDA not available")
        return False
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA verified, gpu detected: {gpu_name}")
    return True

if not check_environment():
    exit()

RAW_DATA_DIR = "data/raw"
VECTOR_STORE_PATH = "data/vector_store_faiss"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def run_ingestion():
    print(f"\nstarting ingestion with GPU and model: {EMBEDDING_MODEL_NAME}")
    
    loader = DirectoryLoader(RAW_DATA_DIR, glob="**/*.txt", show_progress=True, use_multithreading=True)
    docs = loader.load()
    if not docs:
        print(f"no docs found in '{RAW_DATA_DIR}'")
        return
    print(f"loading {len(docs)} docs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"docs divided in {len(chunks)} chunks.")

    print("loading model embedding for GPU")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("loaded model")

    print("creating vector DB faiss")
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"removing old vector in '{VECTOR_STORE_PATH}'...")
        shutil.rmtree(VECTOR_STORE_PATH)

    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)

    print("-" * 50)
    print("pipeline ingestion conclude")
    print("-" * 50)

if __name__ == "__main__":
    run_ingestion()