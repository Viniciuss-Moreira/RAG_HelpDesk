from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="query for RAG")
    top_k: int = Field(3, description="number of relevants docs to be retrivied", ge=1, le=10)

class SourceChunk(BaseModel):
    page_content: str
    source: str = Field(description="file path")

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]