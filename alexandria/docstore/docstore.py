from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from handler.chunkify import get_document_chunks

from models.document import MultipleDocuments, SingleDocument

class DocStore(ABC):
    async def upsert(
            self, 
            documents: List[SingleDocument], 
            session_id: str,
            transient: bool,
            chunk_token_len: Optional[int] = None
    ) -> MultipleDocuments:
        bundle = await self.squash(documents, session_id, transient)
        bundle.contents = get_document_chunks(bundle.contents, chunk_token_len)
        assert isinstance(bundle.contents, List)
        _bundle = await self._upsert(bundle)
        if bundle.theme != _bundle.theme:
            raise ValueError("expected session id not matched when upserting")
        return _bundle
    
    @abstractmethod
    async def _upsert(
            self,
            multi_docs: MultipleDocuments
    ) -> Dict[str, List[str]]:
        raise NotImplementedError

    async def squash(
            self,
            documents: List[SingleDocument],
            session_id: str,
            transient: bool
    ) -> MultipleDocuments:
        # TODO
        # consider if two files are the same document (but possible different versions)
        if transient:
            return MultipleDocuments(theme=session_id,
                                     contents=documents)
        return await self._squash(documents, session_id)

    @abstractmethod
    async def _squash(
            self,
            documents: List[SingleDocument],
            session_id: str
    ) -> MultipleDocuments:
        raise NotImplemented