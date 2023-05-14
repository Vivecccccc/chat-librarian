from typing import List

from fastapi import APIRouter, Request, UploadFile
from alexandria.vectorstore.router import get_vecstore
from handler.utils import hash_int
from models.document import MultipleDocuments
from models.api import UpsertResponse
from handler.embedding.vectorize import MockVectorize
from handler.file_handler import get_document_from_file
from alexandria.vectorstore.vectorstore import VectorStore
from alexandria.vectorstore.providers.faissvectorstore import FaissVectorStore
from alexandria.docstore.docstore import DocStore
from alexandria.docstore.router import get_docstore
from server.constants import VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN, VECTORSTORE_DOC_SAVE_ROOT_FOR_USER
from server.utils import get_user_belongings

file_router = APIRouter()

@file_router.post(
    "/upsert-file",
    response_model=UpsertResponse
)
async def upsert_file(
    request: Request,
    files: List[UploadFile]
):
    session_id = hash_int(request.cookies.get("cookie"))
    transient = True
    user, holdings = get_user_belongings(request)
    if user.username == "admin":
        transient = False
    documents = []
    for file in files:
        document = await get_document_from_file(file)
        documents.append(document)
    if len(documents) == 0:
        raise
    if "_docstore" not in holdings:
        _docstore = await get_docstore()
        holdings.update({"_docstore": _docstore})
    docstore = holdings.get("_docstore")
    assert isinstance(docstore, DocStore)
    bundle = await docstore.upsert(documents, session_id, transient)
    assert isinstance(bundle, MultipleDocuments)
    if len(bundle.contents) == 0:
        return UpsertResponse(ids=[])
    # TODO add a vector store router; test if restore_from should be specified
    if "_vecstore" not in holdings:
        _vecstore = get_vecstore(session_id=session_id,
                                 transient=transient,
                                 )
        holdings.update({"_vecstore": _vecstore})
        #  TODO add a vectorizing router
    vectorize = MockVectorize()
    vecstore = holdings.get("_vecstore")
    assert isinstance(vecstore, VectorStore)
    await vecstore.upsert(bundle, vectorize)
    vectorstore_save_root = VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN
    if transient:
        vectorstore_save_root = VECTORSTORE_DOC_SAVE_ROOT_FOR_USER % (str(session_id))
    if isinstance(vecstore, FaissVectorStore):
        await vecstore.serializing(save_root=vectorstore_save_root,
                              is_doc=True)
    else:
        await vecstore.serializing(save_root=vectorstore_save_root)
    bundle_ids = [x.doc_id for x in bundle.contents]
    return UpsertResponse(ids=bundle_ids)