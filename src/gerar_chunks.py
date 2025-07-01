import json
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuração ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CHUNKS_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "chunks.json")

def generate_chunks_only():
    """
    Executa apenas a parte de carregamento e divisão de texto para
    gerar o arquivo `chunks.json` necessário para o HybridReranker.
    NÃO GERA NOVOS EMBEDDINGS.
    """
    print("Iniciando a geração do arquivo 'chunks.json'...")

    # Garante que o diretório de saída exista
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 1. Carregar documentos do diretório raw
    print(f"Carregando documentos de '{RAW_DATA_DIR}'...")
    loader = DirectoryLoader(RAW_DATA_DIR, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    
    # 2. Dividir em chunks usando a mesma lógica
    # É crucial que chunk_size e chunk_overlap sejam os mesmos usados
    # quando você gerou seu arquivo .npy original.
    print("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    # Converte os documentos do LangChain para uma lista de dicionários
    chunks_list_of_dicts = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in split_docs
    ]
    print(f"Criados {len(chunks_list_of_dicts)} chunks.")

    # 3. Salva os chunks em formato JSON
    print(f"Salvando chunks em '{CHUNKS_OUTPUT_PATH}'...")
    with open(CHUNKS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_list_of_dicts, f, ensure_ascii=False, indent=4)

    print("-" * 50)
    print("Operação concluída com sucesso!")
    print(f"Arquivo '{CHUNKS_OUTPUT_PATH}' foi criado.")
    print("Seu arquivo de embeddings .npy NÃO foi modificado.")
    print("-" * 50)

if __name__ == "__main__":
    generate_chunks_only()