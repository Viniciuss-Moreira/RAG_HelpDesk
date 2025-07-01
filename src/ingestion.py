import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importação moderna para os embeddings, evitando avisos
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuração ---
# Caminhos relativos a partir da raiz do projeto
RAW_DATA_DIR = "data/raw"
VECTOR_STORE_PATH = "data/vector_store_faiss"

# Modelo de embedding correto e multilíngue (1024 dimensões)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def run_ingestion():
    """
    Executa o pipeline de ingestão de dados usando os componentes mais
    recentes do LangChain para criar um Vector Store FAISS.
    """
    print(f"Iniciando ingestão com o modelo de embedding: {EMBEDDING_MODEL_NAME}")

    # 1. Carregar documentos
    print(f"Carregando documentos de '{RAW_DATA_DIR}'...")
    loader = DirectoryLoader(RAW_DATA_DIR, glob="**/*.txt", show_progress=True, use_multithreading=True)
    docs = loader.load()
    
    if not docs:
        print(f"AVISO: Nenhum documento encontrado em '{RAW_DATA_DIR}'. A ingestão será interrompida.")
        return
    print(f"Carregados {len(docs)} documentos.")

    # 2. Dividir documentos em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documentos divididos em {len(chunks)} chunks.")

    # 3. Configurar o modelo de embedding
    print("Carregando o modelo de embedding (isso pode levar um tempo na primeira vez)...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Mude para 'cuda' se tiver GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Modelo de embedding carregado.")

    # 4. Criar e salvar o Vector Store
    print("Criando o banco de dados vetorial FAISS...")
    
    # Apaga o diretório antigo para garantir que não haja conflito de dimensões
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Removendo o Vector Store antigo em '{VECTOR_STORE_PATH}'...")
        shutil.rmtree(VECTOR_STORE_PATH)
        print("Vector Store antigo removido.")

    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)

    print("-" * 50)
    print("Pipeline de ingestão concluído com sucesso!")
    print(f"Vector Store salvo em: '{VECTOR_STORE_PATH}'")
    print("-" * 50)


if __name__ == "__main__":
    run_ingestion()