from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from alexandria.vectorstore.router import get_vecstore
from handler.embedding.router import get_vectorize
from handler.utils import hash_int
from models.document import MultipleDocuments, SingleDocument
from models.api import Settings, UpsertResponse
from handler.file_handler import get_document_from_file
from alexandria.vectorstore.vectorstore import VectorStore
from alexandria.docstore.docstore import DocStore
from alexandria.docstore.router import get_docstore
from server.constants import VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN, VECTORSTORE_DOC_SAVE_ROOT_FOR_USER
from server.utils import get_user_belongings_from_cookies

file_router = APIRouter()

@file_router.post(
    "/upsert-file",
    response_model=UpsertResponse
)
async def upsert_file(
    request: Request,
    files: List[UploadFile]
):
    cookies = request.cookies
    if not cookies:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="not authorized or invalid cookies")
    session_id = hash_int(cookies.get("stage1"))
    transient = True
    user, holdings = get_user_belongings_from_cookies(cookies)
    if user.username == "admin":
        transient = False
    _settings = holdings.get("settings", None)
    if _settings is None or not isinstance(_settings, Settings):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="incorrect configuration")
    mode = _settings.mode
    if mode == "query-only":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="query mode permits no file upserting")
    chunk_size = _settings.chunk_size
    documents, docstore = await _init_docstore(session_id,
                                               transient,
                                               files,
                                               holdings)
    bundle = await docstore.upsert(documents, session_id, transient, chunk_size)
    assert isinstance(bundle, MultipleDocuments)
    if len(bundle.contents) == 0:
        return UpsertResponse(ids=[])
    restore_root = VECTORSTORE_DOC_SAVE_ROOT_FOR_USER % (str(session_id)) if transient \
    else VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN
    vecstore, vectorize = await _init_vecstore(session_id, 
                                               transient, 
                                               holdings, 
                                               _settings)
    await vecstore.upsert(bundle, vectorize)
    await vecstore.serializing(save_root=restore_root, is_doc=True)
    bundle_ids = [x.doc_id for x in bundle.contents]
    return UpsertResponse(ids=bundle_ids)

async def _init_vecstore(session_id: int, 
                         transient: bool, 
                         holdings: Dict[str, Any], 
                         settings: Settings):
    vectorstore = settings.vectorstore
    restore_root = VECTORSTORE_DOC_SAVE_ROOT_FOR_USER % (str(session_id)) if transient \
    else VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN
    if "_vecstore" not in holdings:
        _vecstore = await get_vecstore(session_id=session_id,
                                       transient=transient,
                                       vecstore=vectorstore,
                                       restore_root=restore_root,
                                       dim=512)
        holdings.update({"_vecstore": _vecstore})
    vecstore = holdings.get("_vecstore")
    assert isinstance(vecstore, VectorStore)
    vectorize = await get_vectorize(settings)
    return vecstore, vectorize

async def _init_docstore(session_id: str,
                         transient: bool,
                         files: List[UploadFile], 
                         holdings: Dict[str, Any]) -> tuple[List[SingleDocument], DocStore]:
    documents: List[SingleDocument] = []
    for file in files:
        document = await get_document_from_file(file)
        documents.append(document)
    if len(documents) == 0:
        raise
    if "_docstore" not in holdings:
        _docstore = await get_docstore(session_id=session_id,
                                       transient=transient)
        holdings.update({"_docstore": _docstore})
    docstore = holdings.get("_docstore")
    assert isinstance(docstore, DocStore)
    return documents, docstore