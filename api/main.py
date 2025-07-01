from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import QueryRequest, QueryResponse, SourceChunk
from .rag_chain import get_rag_chain

# --- Aplicação FastAPI e Carregamento do Pipeline RAG ---
app = FastAPI(
    title="Helpdesk RAG API",
    description="API para responder perguntas sobre uma base de conhecimento de TI.",
    version="1.0.0"
)

# Carrega a cadeia RAG uma única vez na inicialização da API
# Isso é essencial para a performance, evitando recarregar modelos a cada requisição.
try:
    rag_chain = get_rag_chain()
except Exception as e:
    raise RuntimeError(f"Erro fatal ao carregar o pipeline RAG: {e}")

# Adiciona o middleware de CORS para permitir requisições do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja para a URL do seu front-end
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Endpoints da API ---
@app.get("/", tags=["Status"])
def read_root():
    """Endpoint raiz para verificar se a API está online."""
    return {"status": "API online e funcionando."}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest):
    """
    Recebe uma pergunta e a processa através da cadeia RAG para retornar uma resposta.
    """
    print(f"Processando query: '{request.query}'")
    try:
        # Invoca a cadeia RAG com a pergunta do usuário.
        # A cadeia já está configurada para retornar a resposta e os chunks.
        result = rag_chain.invoke(request.query)

        # Mapeia os documentos LangChain para o nosso schema Pydantic
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
        print(f"Erro durante o processamento da query: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")