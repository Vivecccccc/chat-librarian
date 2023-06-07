from fastapi import HTTPException, status
from pydantic import BaseModel, BaseSettings, root_validator, validator
from typing import List, Optional
from models.document import SingleDocument
# from models.generic import Query, QueryResult

class UpsertRequest(BaseModel):
    documents: List[SingleDocument]

class UpsertResponse(BaseModel):
    ids: List[str]
    urls: List[str]

class QueryRequest(BaseModel):
    query: str
    top_k: int

# class QueryResponse(BaseModel):
#     results: List[QueryResult]

class Settings(BaseSettings):
    mode: str
    chunk_size: int
    embedding_method: str
    vectorstore: str
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_version: Optional[str] = None

    @validator("mode", pre=True)
    def check_mode(cls, v):
        if v not in {"query-only", "upsert-and-query"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="set mode not allowed")
        return v
    
    @validator("chunk_size")
    def check_chunk_size(cls, v):
        if v < 50:
            v = 50
            print(f"too small chunk size, forced set to {v}")
        elif v > 800:
            v = 800
            print(f"too large chunk size, forced set to {v}")
        return v
    
    @validator("embedding_method")
    def check_embedding_method(cls, v):
        if v not in {"openai", "standard"}:
            v = "standard"
            print(f"embedding method not allowed, fall back to standard method")
        return v
    
    @validator("vectorstore")
    def check_vectorstore(cls, v):
        if v not in {"FAISS", "naive"}:
            v = "naive"
            print(f"vector store not allowed, fall back to naive storage")
        return v