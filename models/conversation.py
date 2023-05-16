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

    def prompt_for_embedding(self,
                             prompt_template: Dict[str, str] = {}) -> str:
        request_key = prompt_template.get("request", "USER INPUT")
        response_key = prompt_template.get("response", "ASSISTANT RESPONSE")
        context_key = prompt_template.get("context", None)
        prompt = ""
        prompt += f"{context_key}: {self.context}; \n" if context_key and self.context else ""
        prompt += f"{request_key}: {self.request}; \n" if request_key and self.request else ""
        prompt += f"{response_key}: {self.response}; \n" if self.response else f"{response_key}: "
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

    def to_list(self, 
                existed: List[SingleConversation] = [],
                reverse: bool = False) -> List[SingleConversation]:
        result = [x for x in existed] if existed else []
        current = self
        while current is not None:
            result.append(current.curr_conv)
            if reverse is not True:
                current = current.next_conv
            else:
                current = current.prev_conv
        return result
    
    def update_dict(self,
                    existed: Dict[str, SingleConversation]):
        result = existed
        current = self
        while current is not None:
            result.update({current.curr_conv.conv_id: current.curr_conv})
            current = current.next_conv

class ConversationEmbeddings(BaseModel):
    embeddings: Dict[SingleConversation, Optional[List[float]]]

class MultipleConversation(Bundle):
    theme: Optional[str] = None
    contents: Optional[List[SingleConversation]] = None
    embedding: Optional[ConversationEmbeddings] = None