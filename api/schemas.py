from pydantic import BaseModel
from typing import List, Optional

# Schema para a requisição que chega na API
class QueryRequest(BaseModel):
    question: str

# Schema para um documento fonte retornado
class SourceDocument(BaseModel):
    page_content: str
    metadata: Optional[dict] = None

# Schema para a resposta que a API envia
class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[SourceDocument]] = None