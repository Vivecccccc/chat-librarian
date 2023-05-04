import logging

from pydantic import BaseModel, validator
from typing import List, Optional
from enum import Enum
from datetime import datetime

from models.utils import Metadata

logger = logging.getLogger(__name__)

class DocumentVersion(BaseModel):
    version_url: List[str] = []
    version_hash: List[str] = []
    modified_at: List[datetime] = []

    @validator("modified_at", each_item=True)
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
    

class DocumentMetadata(Metadata):
    created_by: Optional[str] = None
    versions: Optional[DocumentVersion] = None


class SingleDocument(BaseModel):
    doc_id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None

class SingleDocumentWithChunks(SingleDocument):
    chunks: List[DocumentChunk]