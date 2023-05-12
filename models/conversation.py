import logging

from pydantic import BaseModel
from typing import Dict, List, Optional
from handler.utils import hash_int

from models.generic import Bundle, Metadata

logger = logging.getLogger(__name__)

class ConversationMetadata(Metadata):
    created_by: Optional[List[str]] = None

class SingleConversation(BaseModel):
    conv_id: str
    context: Optional[str] = None
    request: Optional[str] = None
    response: Optional[str] = None
    metadata: Optional[ConversationMetadata] = None

    def prompt_for_embedding(self, include_ctx: bool = False) -> str:
        prompt = f"""
            USER INPUT: {self.request}
            ASSISTANT RESPONSE: {self.response}
            """
        if include_ctx:
            prompt = f"""
            RELATED MATERIALS: {self.context}

            """ + prompt
        return prompt

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SingleConversation):
            return self.conv_id == other.conv_id
        return False
    
    def __hash__(self) -> int:
        return hash_int(self.conv_id)

class Conversation(BaseModel):
    curr_conv: Optional[SingleConversation] = None
    next_conv: Optional['Conversation'] = None
    prev_conv: Optional['Conversation'] = None

    def add_next(self, next_conv: Optional['Conversation']):
        self.next_conv = next_conv
        if next_conv is not None:
            next_conv.prev_conv = self

    def add_prev(self, prev_conv: Optional['Conversation']):
        self.prev_conv = prev_conv
        if prev_conv is not None:
            prev_conv.next_conv = self

    def to_list(self) -> List[SingleConversation]:
        result = []
        current = self
        while current is not None:
            result.append(current.curr_conv)
            current = current.next_conv
        return result

class ConversationEmbeddings(BaseModel):
    embeddings: Dict[SingleConversation, Optional[List[float]]]

class MultipleConversation(Bundle):
    theme: Optional[str] = None
    contents: Optional[List[SingleConversation]] = None
    embedding: Optional[ConversationEmbeddings] = None