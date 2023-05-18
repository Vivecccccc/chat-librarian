import logging

from pydantic import BaseModel, validator
from typing import List, Any, Optional
from enum import Enum
from datetime import datetime
from handler.utils import hash_int


logger = logging.getLogger(__name__)

class Source(str, Enum):
    document = "document"
    conversation = "conversation"

class Metadata(BaseModel):
    source: Optional[Source] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    
    @validator("created_at")
    def _parse_datetime(cls, val):
        if val is None:
            return val
        import arrow
        if isinstance(val, datetime):
            return val.replace(tzinfo=None)
        assert isinstance(val, str), "modified_at is neither in string format nor datetime"
        try:
            time_obj = arrow.get(val)
            return time_obj.datetime.replace(tzinfo=None)
        except arrow.parser.ParserError:
            logger.error(f"Invalid date format: {val}")
            return arrow.now()
        
    def __hash__(self):
        fields = [self.source, self.created_at, self.created_by]
        fields_str = [str(field) if field is not None else '' for field in fields]
        return hash_int(''.join(fields_str))
    
class Bundle(BaseModel):
    theme: Optional[str] = None
    contents: Optional[List[Any]] = None

    def __len__(self):
        if not self.contents:
            return 0
        return len(self.contents)
    
# class Query(BaseModel):
#     query: str
#     filter: Optional[DocumentFilter] = None
#     top_k: Optional[int] = 3

# class QueryResult(BaseModel):
#     query: str
#     documents: List[DocumentChunkWithScore]
#     adhered_conv: Conversation
#     related_conv: SingleConversation