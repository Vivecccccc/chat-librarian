from alexandria.vectorstore.vectorstore import VectorStore
import os

async def get_vecstore(session_id: str,
                       transient: bool,
                       **kwargs) -> VectorStore:
    vecstore = os.environ.get("VECTORSTORE", "NAIVE")
    assert vecstore is not None
    restore_index_from = kwargs.get("restore_index_from", None)
    restore_map_from = kwargs.get("restore_map_from", None)
    match vecstore:
        case "FAISS":
            from alexandria.vectorstore.providers.faissvectorstore import FaissVectorStore
            dim = kwargs.get("dim", None)
            if dim is None:
                raise ValueError("dimension should be specified for FAISS")
            index_key = kwargs.get("index_key", "Flat")
            return FaissVectorStore(dim=dim,
                                    session_id=session_id,
                                    transient=transient,
                                    index_key=index_key,
                                    restore_index_from=restore_index_from,
                                    restore_map_from=restore_map_from)
        case _:
            from alexandria.vectorstore.providers.naivevectorstore import NaiveVectorStore

            return NaiveVectorStore(session_id=session_id,
                                    transient=transient,
                                    restore_index_from=restore_index_from,
                                    restore_map_from=restore_map_from)
