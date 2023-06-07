
from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException, Header, Request, status
from alexandria.chatstore.chatstore import ChatStore
from handler.utils import hash_int

from models.api import QueryRequest, Settings
from server.utils import get_user_belongings


conversation_router = APIRouter()

@conversation_router.post(
    "/query",
)
async def query(
    base_request: Request,
    request: QueryRequest = Body(...)
):
    cookies = base_request.cookies
    if not cookies:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="not authorized or invalid cookies")
    session_id = hash_int(cookies.get("stage1"))
    user, holdings = get_user_belongings(request=base_request)
    _settings = holdings.get("settings", None)
    if _settings is None or not isinstance(_settings, Settings):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="incorrect configuration")
    mode = _settings.mode
    transient = True if mode == "upsert-and-query" else False
    chatstore = _init_chatstore(session_id=session_id, transient=transient, holdings=holdings, settings=_settings)
    q = request.query
    messages, srcs = await chatstore.eloquence(q)
    response = await chatstore.chat(msgs=messages)
    await chatstore.echo_response((q, response))
    return {'msg': response, 'src': [s.dict() for s in srcs]}

def _init_chatstore(session_id: int,
                    transient: bool,
                    holdings: Dict[str, Any],
                    settings: Settings):
    if "_chatstore" not in holdings:
        _chatstore = ChatStore(session_id=session_id,
                               transient=transient,
                               holdings=holdings,
                               settings=settings)
        holdings.update({"_chatstore": _chatstore})
    chatstore = holdings.get("_chatstore")
    assert isinstance(chatstore, ChatStore)
    return chatstore
