import os
import json
import faiss
import numpy as np
from typing import Dict, List, Optional
from alexandria.vectorstore.vectorstore import VectorStore
from models.conversation import MultipleConversation, SingleConversation
from models.document import DocumentChunkWithEmbedding, SingleDocumentWithChunks
from models.generic import Bundle

class FaissVectorStore(VectorStore):
    ALLOWED_INDEX_TYPE = {
        "Flat",
    }
    def __init__(self,
                 dim: int,
                 session_id: int,
                 transient: bool,
                 index_key: str = "Flat",
                 restore_index_from: Optional[str] = None,
                 restore_map_from: Optional[str] = None,
                 cuda: bool = False,
        ):
        """
        Initializes a new instance of the FaissVectorStore class.

        Args:
        - dim: An integer representing the dimensionality of the embeddings.
        - session_id: A string representing the session ID.
        - transient: A boolean flag indicating whether the embeddings should be stored permanently or temporarily.
        - index_key: A string representing the type of index to use (allowed values: "Flat").
        - restore_index_from: An optional string representing the path to a previously saved index.
        - restore_map_from: An optional string representing the path to a previously saved map.
        - cuda: A boolean flag indicating whether to use GPU for computations.
        """
        self.d: int = dim
        self.session_id: int = session_id
        self.transient: bool = transient
        self.index_key: str = index_key
        self.restore_index_from: Optional[str] = restore_index_from
        self.restore_map_from: Optional[str] = restore_map_from
        self.cuda: bool = cuda
        self.index: Optional[faiss.Index] = None
        self.map: Optional[Dict[str, List[str]]] = None
        self.device = None
        try:
            self._setup_index()
        except Exception as e:
            print(f"Initializing index failure: {e}")
        self._setup_map()
    
    def _setup_map(self):
        """
        Sets up the map instance variable, which is a dictionary that stores the mapping between document or
        conversation IDs and their corresponding chunk IDs. If a path to a previously saved map is provided, it reads
        the map from the file.
        """    
        if self.restore_map_from is not None and os.path.isfile(self.restore_map_from):
            with open(self.restore_map_from, 'r') as f:
                self.map = json.load(f)
        else:
            self.map = {}

    def _setup_index(self):
        """
        Sets up the index instance variable, which is a FAISS index object used to store and retrieve embeddings. If
        a path to a previously saved index is provided, it reads the index from the file. If not, it creates a new index
        using the specified index key. If a GPU is available and cuda is True, it uses the GPU for computations.
        """       
        if self.restore_index_from is not None and os.path.isfile(self.restore_index_from):
            index = faiss.read_index(self.restore_from)
            self.index = index
        else:
            index = faiss.index_factory(self.d, self.index_key)
            if self.cuda and faiss.get_num_gpus() > 0:
                raise NotImplemented
                self.device = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self.device, 0, index)
            if self.index_key == "Flat":
                index = faiss.IndexIDMap2(index)
            self.index = index
        assert self.index.d == self.d, "Initializing index failure: dimension not aligned"     

    def _remove_existed(self, ids: Optional[np.ndarray | List[int]]):
        """
        Removes embeddings with IDs that already exist in the index. The ids parameter can be a list of IDs or a numpy
        array of integer values. If ids is None or an empty list, the method returns 0.
        
        Args:
        - ids: A numpy array or list of integer values representing the IDs to remove from the index.
        
        Returns:
        - An integer representing the number of embeddings removed from the index.
        """
        if ids is None:
            return 0
        if isinstance(ids, List):
            ids = np.asarray(ids, dtype=np.int64)
        if ids.shape[0] == 0:
            return 0
        return self.index.remove_ids(ids)

    def _add(self, vectors: List[List[float]], ids: List[int]):
        """
        Adds embeddings and their corresponding IDs to the index. It converts the input lists to numpy arrays and calls
        the add_with_ids method of the FAISS index object.
        
        Args:
        - vectors: A list of lists of float values representing the embeddings to add to the index.
        - ids: A list of integer values representing the IDs of the embeddings to add to the index.
        """
        vectors: np.ndarray = np.asarray(vectors, dtype=np.float32)
        ids: np.ndarray = np.asarray(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, ids)

    async def _upsert(
            self, 
            bundle: Bundle
    ):
        """
        Updates or inserts embeddings and their corresponding IDs into the index. It extracts the embeddings and their
        IDs from the input bundle, removes the embeddings with versioned IDs that already exist in the index, and adds
        the new embeddings to the index. It also updates the map instance variable with the new IDs.
        
        Args:
        - bundle: A Bundle object representing the embeddings to update or insert into the index.
        """
        session_id = int(bundle.theme)
        if self.transient:
            assert session_id == self.session_id, "session_id not matched a recorded one"
        contents = bundle.contents
        versioned_sub_ids: List[int] = []
        updated_sub_ids: List[int] = []
        updated_embeddings: List[List[float]] = []
        for elem in contents:
            if isinstance(elem, SingleDocumentWithChunks):
                doc_id = elem.doc_id
                subs = elem.chunks
                versioned_sub_ids.extend(self.map.get(doc_id, []))
            elif isinstance(elem, SingleConversation):
                subs = [elem]
            else:
                raise ValueError
            for sub in subs:
                if isinstance(sub, DocumentChunkWithEmbedding):
                    updated_sub_ids.append(sub.chunk_id)
                    updated_embeddings.append(sub.embedding)
                elif isinstance(sub, SingleConversation):
                    updated_sub_ids.append(hash(sub))
                    assert isinstance(bundle, MultipleConversation)
                    updated_embeddings.append(bundle.embedding.embeddings.get(sub))
        existed_cnt = self._remove_existed(versioned_sub_ids)
        print(f"removed found {existed_cnt} existed id(s)")
        self._add(updated_embeddings, updated_sub_ids)

    async def _query(self, vectors: List[List[float]], k: int = 3) -> List[List[int]]:
        """
        Query the index to find the k most similar records to the input vectors.
        Args:
            vectors: A list of vectors to query the index with.
            k: The number of most similar records to return.
        Returns:
            A list of the IDs of the k most similar records for each query vector.
        """
        vectors: np.ndarray = np.asarray(vectors, dtype=np.float32)
        _, idx = self.index.search(vectors, k)
        return idx.tolist()