import os
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional
from alexandria.vectorstore.vectorstore import VectorStore
from models.conversation import MultipleConversation, SingleConversation
from models.document import DocumentChunkWithEmbedding, SingleDocumentWithChunks
from models.generic import Bundle

class NaiveVectorStore(VectorStore):
    def __init__(self,
                 session_id: int,
                 transient: bool,
                 restore_index_from: Optional[str] = None,
                 restore_map_from: Optional[str] = None
                 ):
        self.session_id: int = session_id
        self.transient: bool = transient
        self.restore_index_from: Optional[str] = restore_index_from
        self.restore_map_from: Optional[str] = restore_map_from
        self.raw_storage: Optional[Dict[int, List[float]]] = None
        self.doc_map: Optional[Dict[str, List[str]]] = None
        self.arr_storage: Optional[np.ndarray] = None
        self.stored_ids: Optional[List[int]] = None
        self.has_queried_since_update: bool = False
        self._setup_index()
        self._setup_doc_map()

    def _setup_doc_map(self):
        if self.restore_map_from is not None and os.path.isfile(self.restore_map_from):
            import json
            with open(self.restore_map_from, 'r') as f:
                self.doc_map = json.load(f)
        else:
            self.doc_map = {}

    def _setup_index(self):
        self.raw_storage: Optional[Dict[int, List[float]]] = {}
        self.storage = None
        self.stored_ids = None
        if self.restore_index_from is not None:
            if os.path.isfile(self.restore_index_from):
                import json
                with open(self.restore_index_from, 'r') as f:
                    self.raw_storage: Dict[int, List[float]] = json.load(f)
        

    def _remove_existed(self, ids: Optional[List[int]]) -> int:
        if self.raw_storage is None:
            raise ValueError("raw storage (dict-like) not initialized")
        if not ids:
            return 0
        cnt = 0
        for id in ids:
            _ = self.raw_storage.pop(id, None)
            if _:
                cnt += 1
        if cnt != 0:
            self.has_queried_since_update = False
        return cnt
    
    def _add(self, vectors: List[List[float]], ids: List[int]):
        if self.raw_storage is None:
            raise ValueError("raw storage (dict-like) not initialized")
        assert len(vectors) == len(ids), "vectors and ids to be inserted not aligned"
        for id, vector in zip(ids, vectors):
            self.raw_storage.update({id: vector})
        self.has_queried_since_update = False

    def __cosine_similarity(self, q: np.ndarray, v: np.ndarray) -> float:
        return 1 - cosine(q, v)
    
    def _find_topk(self, query: np.ndarray, k: int):
        candidates = self.storage
        ids = self.stored_ids
        if candidates is None:
            raise ValueError("storage (ndarray-like) not initialized")
        similarities = np.apply_along_axis(lambda v: self.__cosine_similarity(query, v),
                                           axis=1,
                                           arr=candidates)
        indices = np.argpartition(-similarities, k)[:k]
        _subset = sorted([(i, similarities[i]) for i in indices], key=lambda x: x[1], reverse=True)
        return [ids[x[0]] for x in _subset]

    async def _upsert(self, bundle: Bundle):
        session_id = int(bundle.theme)
        if self.transient:
            assert session_id == self.session_id
        contents = bundle.contents
        versioned_sub_ids: List[int] = []
        updated_sub_ids: List[int] = []
        updated_embeddings: List[List[float]] = []
        for elem in contents:
            if isinstance(elem, SingleDocumentWithChunks):
                doc_id = elem.doc_id
                subs = elem.chunks
                versioned_sub_ids.extend(self.doc_map.get(doc_id, []))
                self.doc_map.update({doc_id: []})
            elif isinstance(elem, SingleConversation):
                subs = [elem]
            else:
                raise ValueError
            for sub in subs:
                if isinstance(sub, DocumentChunkWithEmbedding):
                    self.doc_map.get(doc_id).append(sub.chunk_id)
                    updated_sub_ids.append(sub.chunk_id)
                    updated_embeddings.append(sub.embedding)
                elif isinstance(sub, SingleConversation):
                    updated_sub_ids.append(hash(sub))
                    assert isinstance(bundle, MultipleConversation)
                    updated_embeddings.append(bundle.embedding.embeddings.get(sub))
        existed_cnt = self._remove_existed(versioned_sub_ids)
        print(f"removed found {existed_cnt} existed id(s)")
        self._add(updated_embeddings, updated_sub_ids)

    async def _query(self, vectors: List[List[float]], k: int = 3):
        if not self.raw_storage:
            raise ValueError("raw storage (dict-like) not initialized")
        if self.has_queried_since_update is False:
            self.storage = np.array(list(self.raw_storage.values()), dtype=np.float32)
            self.stored_ids = list(self.raw_storage.keys())
            self.has_queried_since_update = True
        candidates: List[List[int]] = []
        queries = np.array(vectors, dtype=np.float32)
        for query in queries:
            candidates.append(self._find_topk(query, k))
        return candidates
    
    async def serializing(self, save_root: str, is_doc: bool):
        os.makedirs(save_root, exist_ok=True)
        index_save_to = os.path.join(save_root, "vectors.json")
        import json
        if self.raw_storage:
            try:
                with open(index_save_to, 'w') as f:
                    json.dump(self.raw_storage, f)
                print(f"JSON index written to {index_save_to}")
            except Exception as e:
                print(f"JSON index saving to {index_save_to} failed")
                raise e
        else:
            raise ValueError("JSON index has not been initialized")
        if is_doc:
            if self.doc_map:
                map_save_to = os.path.join(save_root, "mappings.json")
                with open(map_save_to, 'w') as f:
                    json.dump(self.doc_map, f)
                print(f"document ID mapping written to {map_save_to}")
            else:
                print(f"document mapping has not been initialized")