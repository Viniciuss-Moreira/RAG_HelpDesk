from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """Schema para a requisição da query."""
    query: str = Field(..., description="A pergunta do usuário para o RAG.")
    top_k: int = Field(3, description="Número de documentos relevantes a serem recuperados.", ge=1, le=10)

class SourceChunk(BaseModel):
    """Schema para um chunk de documento fonte."""
    page_content: str
    source: str = Field(description="Caminho do arquivo de origem.")

class QueryResponse(BaseModel):
    """Schema para a resposta da API."""
    answer: str
    source_chunks: List[SourceChunk]