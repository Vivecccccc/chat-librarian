from abc import ABC
from typing import List
from handler.embedding.vectorize import Vectorize, embed_bundle

from models.generic import Bundle

class VectorStore(ABC):
    async def upsert(
            self,
            bundle: Bundle,
            emb_method: Vectorize
    ):
        _bundle = await embed_bundle(bundle, emb_method)
        await self._upsert(_bundle)

    async def _upsert(
            self,
            bundle: Bundle
    ):
        raise NotImplemented
        
    async def query(
            self,
            texts: List[str],
            emb_method: Vectorize
    ):
        q_emb = await emb_method.embed_text_bundle(texts)
        await self._query(q_emb, k=3)

    async def _query(
            self,
            vectors: List[List[float]],
            k: int = 3
    ):
        raise NotImplemented
    

                