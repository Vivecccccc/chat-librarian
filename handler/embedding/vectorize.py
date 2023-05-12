from abc import ABC, abstractmethod
from typing import List
from models.conversation import ConversationEmbeddings, MultipleConversation, SingleConversation
from models.document import DocumentChunkWithEmbedding, MultipleDocuments, SingleDocumentWithChunks

from models.generic import Bundle

class Vectorize(ABC):
    @abstractmethod
    async def embed_text_bundle(
            self,
            text: List[str]
    ) -> List[List[float]]:
        raise NotImplemented
    
    @abstractmethod
    async def embed_text(
        self,
        text: str
    ) -> List[float]:
        raise NotImplemented
    
async def embed_bundle(
        bundle: Bundle,
        emb_method: Vectorize
) -> Bundle:
    contents = bundle.contents
    assert contents is not None
    _generated = []
    for elem in contents:
        if isinstance(elem, SingleDocumentWithChunks):
            texts = [chunk.text for chunk in elem.chunks]
            embedding = await emb_method.embed_text_bundle(texts)
            assert len(embedding) == len(elem.chunks)
            _chunks = [DocumentChunkWithEmbedding(**chunk.dict(), embedding=emb) 
                        for chunk, emb 
                        in zip(elem.chunks, embedding)]
            _generated.append(SingleDocumentWithChunks(**elem.dict(exclude={"chunks"}), chunks=_chunks))
        elif isinstance(elem, SingleConversation):
            text = elem.prompt_for_embedding(include_ctx=False)
            embedding = emb_method.embed_text(text)
            _generated.append((elem, embedding))
        else:
            raise ValueError
    if isinstance(bundle, MultipleDocuments):
        return MultipleDocuments(theme=bundle.theme,
                                 contents=_generated)
    elif isinstance(bundle, MultipleConversation):
        conv_emb = ConversationEmbeddings(embeddings={k: v for k, v in _generated})
        return MultipleConversation(theme=bundle.theme,
                                    contents=contents,
                                    embedding=conv_emb)
    else:
        raise ValueError

class MockVectorize(Vectorize):
    async def embed_text_bundle(self, text: List[str]) -> List[List[float]]:
        import numpy as np
        _arr = np.random.randn(len(text), 512).tolist()
        return _arr
    
    async def embed_text(self, text: str) -> List[float]:
        import numpy as np
        return np.random.randn(512).tolist()