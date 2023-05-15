from alexandria.docstore.docstore import DocStore
import os

from alexandria.docstore.providers.jsondocstore import JsonDocStore


async def get_docstore(session_id: str,
                       transient: bool) -> DocStore:
    return JsonDocStore(session_id=session_id, transient=transient)
    # datastore = os.environ.get("DATASTORE", "JSON")
    # assert datastore is not None

    # match datastore:
    #     case "redis":
    #         #TODO
    #         raise NotImplemented
    #     case _:
    #         from alexandria.docstore.providers.jsondocstore import JsonDocStore
            
    #         return JsonDocStore(storage_root='.data')
