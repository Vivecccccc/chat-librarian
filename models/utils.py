import logging

from pydantic import BaseModel, validator
from typing import *
from abc import ABC
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class Source(str, Enum):
    document = "document"
    conversation = "conversation"

class Metadata(BaseModel):
    source: Optional[Source] = None
    source_id: Optional[str] = None
    created_at: Optional[datetime] = None
    
    @validator("created_at")
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
