import sys
from pathlib import Path
import traceback

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import QueryRequest, QueryResponse, SourceChunk
from api.rag_chain import get_rag_chain


app = FastAPI(
    title="Helpdesk RAG API",
    description="API for answering questions about an IT knowledge base",
    version="1.0.0"
)

try:
    rag_chain = get_rag_chain()
except Exception as e:
    print("error to load pipeline RAG")
    traceback.print_exc()
    raise RuntimeError(f"error to load pipeline RAG: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", tags=["Status"])
def read_root():
    return {"status": "API ON"}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest):
    print(f"processing query: '{request.query}'")
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
        print(f"error to process query")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"internal error. check the server console")