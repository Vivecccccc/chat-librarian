from pydantic import BaseModel
from typing import List, Optional
from models.document import SingleDocument
# from models.generic import Query, QueryResult

class UpsertRequest(BaseModel):
    documents: List[SingleDocument]

class UpsertResponse(BaseModel):
    ids: List[str]

# class QueryRequest(BaseModel):
#     queries: List[Query]

# class QueryResponse(BaseModel):
#     results: List[QueryResult]