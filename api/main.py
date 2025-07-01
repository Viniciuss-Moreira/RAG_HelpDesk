import sys
from pathlib import Path
import traceback  # <-- 1. IMPORTAMOS O MÓDULO DE TRACEBACK

# Adiciona a pasta raiz do projeto ao sys.path.
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Corrigindo o import dos schemas para o caminho relativo correto
from api.schemas import QueryRequest, QueryResponse, SourceChunk
from api.rag_chain import get_rag_chain


# --- Aplicação e Carregamento do Pipeline RAG ---
app = FastAPI(
    title="Helpdesk RAG API",
    description="API para responder perguntas sobre uma base de conhecimento de TI.",
    version="1.0.0"
)

try:
    rag_chain = get_rag_chain()
except Exception as e:
    # Imprime o traceback completo se a falha for na inicialização
    print("!!!!!! ERRO FATAL AO CARREGAR O PIPELINE RAG !!!!!!")
    traceback.print_exc()
    raise RuntimeError(f"Erro fatal ao carregar o pipeline RAG: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Endpoints da API ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "API online e funcionando."}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest):
    print(f"Processando query: '{request.query}'")
    try:
        result = rag_chain.invoke(request.query)
        source_chunks = [
            SourceChunk(
                page_content=doc.page_content,
                source=doc.metadata.get('source', 'desconhecida')
            ) for doc in result['source_chunks']
        ]
        return QueryResponse(
            answer=result['answer'],
            source_chunks=source_chunks
        )
    except Exception as e:
        # ####################################################################
        # ## MUDANÇA PRINCIPAL: IMPRIMINDO O TRACEBACK COMPLETO             ##
        # ####################################################################
        print(f"!!!!!! OCORREU UM ERRO DURANTE O PROCESSAMENTO DA QUERY !!!!!!")
        # Esta linha vai imprimir o "mapa" completo do erro no terminal
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno. Verifique o console do servidor para detalhes.")