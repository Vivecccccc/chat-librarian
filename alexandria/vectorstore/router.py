from alexandria.vectorstore.vectorstore import VectorStore
import os

def get_vecstore(session_id: str,
                 transient: bool,
                 vecstore: str,
                 **kwargs) -> VectorStore:
    assert vecstore is not None
    restore_root = kwargs.get("restore_root", None)
    match vecstore:
        case "FAISS":
            from alexandria.vectorstore.providers.faissvectorstore import FaissVectorStore
            dim = kwargs.get("dim", None)
            if dim is None:
                raise ValueError("dimension should be specified for FAISS")
            index_key = kwargs.get("index_key", "Flat")
            restore_index_from = os.path.join(restore_root, "vectors.index") if restore_root else None
            restore_map_from = os.path.join(restore_root, "mappings.json") if restore_root else None
            return FaissVectorStore(dim=dim,
                                    session_id=session_id,
                                    transient=transient,
                                    index_key=index_key,
                                    restore_index_from=restore_index_from,
                                    restore_map_from=restore_map_from)
        case _:
            from alexandria.vectorstore.providers.naivevectorstore import NaiveVectorStore
            restore_index_from = os.path.join(restore_root, "vectors.json") if restore_root else None
            restore_map_from = os.path.join(restore_root, "mappings.json") if restore_root else None
            return NaiveVectorStore(session_id=session_id,
                                    transient=transient,
                                    restore_index_from=restore_index_from,
                                    restore_map_from=restore_map_from)
