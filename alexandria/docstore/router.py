from alexandria.docstore.docstore import DocStore
import os


async def get_docstore() -> DocStore:
    datastore = os.environ.get("DATASTORE", "JSON")
    assert datastore is not None

    match datastore:
        case "redis":
            #TODO
            raise NotImplemented
        case _:
            from alexandria.docstore.providers.jsondocstore import JsonDocStore
            
            return JsonDocStore(storage_root='.data')
