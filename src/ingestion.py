import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (HUGGINGFACE_API_TOKEN)
load_dotenv()

# --- Configuração ---
RAW_DATA_DIR = "data/raw"
VECTOR_STORE_PATH = "data/vector_store_faiss"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def run_ingestion():
    """
    Executa o pipeline de ingestão de dados usando LangChain.
    1. Carrega documentos de um diretório.
    2. Divide os documentos em chunks.
    3. Gera embeddings para os chunks.
    4. Cria e salva um Vector Store FAISS localmente.
    """
    print("Iniciando ingestão com LangChain...")

    # Carrega os documentos do diretório especificado
    loader = DirectoryLoader(RAW_DATA_DIR, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"Carregados {len(docs)} documentos.")

    # Divide os documentos em pedaços menores (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Documentos divididos em {len(chunks)} chunks.")

    # Configura o modelo de embedding que transformará texto em vetores
    model_kwargs = {'device': 'cpu'}  # Use 'cuda' se tiver GPU
    encode_kwargs = {'normalize_embeddings': True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Cria o Vector Store usando os chunks e o modelo de embedding
    print("Criando o vector store com FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings_model)

    # Salva o Vector Store no disco para ser usado pela API
    vector_store.save_local(VECTOR_STORE_PATH)

    print("-" * 50)
    print(f"Ingestão concluída! Vector Store salvo em '{VECTOR_STORE_PATH}'")
    print("-" * 50)


if __name__ == "__main__":
    # Verifica se o token da Hugging Face está configurado
    if not os.getenv("HUGGINGFACE_API_TOKEN"):
        raise ValueError("A variável de ambiente HUGGINGFACE_API_TOKEN não está configurada.")
    run_ingestion()