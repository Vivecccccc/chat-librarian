import logging

from pydantic import BaseModel, validator, root_validator
from typing import List, Optional, Dict
from datetime import datetime

from models.generic import Metadata

logger = logging.getLogger(__name__)

class DocumentVersion(BaseModel):
    version_id: Optional[str] = None
    version_url: Optional[str] = None
    modified_at: Optional[datetime] = None

    @validator("modified_at")
    def _parse_datetime(cls, val):
        import arrow
        if isinstance(val, datetime):
            return val
        assert isinstance(val, str), "modified_at is neither in string format nor datetime"
        try:
            time_obj = arrow.get(val)
            return time_obj.datetime
        except arrow.parser.ParserError:
            logger.error(f"Invalid date format: {val}")
            return arrow.now()
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, DocumentVersion):
            return self.version_id == other.version_id
        return False
    
    def __hash__(self) -> int:
        return hash(self.version_id)
    
    def __le__(self, other: object) -> bool:
        if isinstance(other, DocumentVersion):
            return self.modified_at <= other.modified_at
        return NotImplemented
    
    def __lt__(self, other: object) -> bool:
        if isinstance(other, DocumentVersion):
            return self.modified_at < other.modified_at
        return NotImplemented
    

class DocumentMetadata(Metadata):
    version: Optional[DocumentVersion] = None

class DocumentChunkMetadata(BaseModel):
    doc_id: str
    doc_metadata: DocumentMetadata

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentChunkMetadata

class DocumentChunkWithEmbedding(DocumentChunk):
    embedding: Optional[List[float]] = None

class DocumentChunkWithScore(DocumentChunk):
    score: float
    
class SingleDocument(BaseModel):
    doc_id: str
    text: str
    metadata: DocumentMetadata

class SingleDocumentWithChunks(SingleDocument):
    chunks: List[DocumentChunk]

class ArchivedVersions(BaseModel):
    doc_id: str
    versions: Dict[DocumentVersion, SingleDocument]

    @root_validator
    def _all_ver_same_doc_id(cls, val):
        doc_id, versions = val.get("doc_id"), val.get("versions")
        docs = versions.values()
        for doc in docs:
            if doc.doc_id != doc_id:
                raise ValueError
        return val
    
class DocumentFilter(BaseModel):
    doc_ids: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    final_date: Optional[datetime] = None

class MultipleDocuments(BaseModel):
    theme: Optional[str] = None
    docs: Optional[List[SingleDocument]] = None
