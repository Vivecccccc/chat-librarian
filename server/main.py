import sys
from pathlib import Path

from alexandria.vectorstore.vectorstore import VectorStore
sys.path[0] = str(Path(sys.path[0]).parent)
from alexandria.vectorstore.providers.faissvectorstore import FaissVectorStore
from handler.embedding.openai import OpenAIEmbeddings
from handler.embedding.vectorize import MockVectorize, embed_bundle
from handler.utils import hash_int
from alexandria.docstore.docstore import DocStore
from alexandria.docstore.router import get_docstore
from models.document import MultipleDocuments

from datetime import datetime
from fastapi import FastAPI, File, Request, HTTPException, Depends, Response, UploadFile
from pydantic import BaseModel
from typing import Any, List, Optional
from fastapi import Depends, HTTPException, status, FastAPI
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError, ExpiredSignatureError
from datetime import timedelta, datetime
from typing import Optional, Dict
from passlib.context import CryptContext
from pydantic import BaseModel
from handler.file_handler import get_document_from_file

from models.api import UpsertResponse

STAGE1_SECRET_KEY = "night_mother"
STAGE2_SECRET_KEY = "what-is-the-meaning-of-life"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

class User(BaseModel):
    username: str
    hashed_password: str
    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.dict().values()))

USER_BELONGINGS: Dict[User, Optional[Dict[str, Any]]] = {}
USER_BASIC = {
    "admin": User(username="admin",
                  hashed_password=pwd_context.hash("admin")),
    "user1": User(username="user1",
                  hashed_password=pwd_context.hash("user1")),
    "user2": User(username="user2",
                  hashed_password=pwd_context.hash("user2"))
}

def authenticate_user(username: str, password: str) -> Optional[User]:
    user = USER_BASIC.get(username, None)
    if user and pwd_context.verify(password, user.hashed_password):
        return user
    return None

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=12)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, key=STAGE2_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user_from_cookies(cookies: Dict[str, str]):
    cookie_timeout_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credential invalidated"
    )
    token_timeout_exception = HTTPException(
        status_code=555,
        detail="Session reached maximum expiration"
    )
    if not cookies:
        raise cookie_timeout_exception
    cookie = cookies["cookie"]
    implicit_token = jwt.decode(token=cookie, key=STAGE1_SECRET_KEY, algorithms=ALGORITHM)["token"]
    try:
        info = jwt.decode(token=implicit_token, key=STAGE2_SECRET_KEY, algorithms=ALGORITHM)
    except ExpiredSignatureError:
        raise token_timeout_exception
    except JWTError as je:
        raise je
    username = info["sub"]
    return USER_BASIC.get(username)

def get_user_belongings(request: Request):
    cookies = request.cookies
    user = get_current_user_from_cookies(cookies)
    is_existed_user = user in USER_BELONGINGS
    if not is_existed_user:
        USER_BELONGINGS.update({user: {}})
    return user, USER_BELONGINGS.get(user)

@app.get("/echo")
async def echo(request: Request, response: Response):
    print(request.cookies)
    holdings = get_user_belongings(request)
    holdings.update({"yes": "2"})
    USER_BELONGINGS.update({get_current_user_from_cookies(request.cookies): {"what": "1"}})
    return "echo"

@app.post("/login")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires_in = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires_in)
    cookie = jwt.encode({"token": access_token}, key=STAGE1_SECRET_KEY, algorithm=ALGORITHM)
    response.set_cookie(key="cookie", value=cookie, max_age=1800)
    return "login successful"

@app.get("/logout")
async def logout(request: Request, response: Response):
    if request.cookies:
        response.delete_cookie(key="cookie")
    return "logout successful"

@app.post(
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
        raise
    if "_vecstore" not in holdings:
        _vecstore = FaissVectorStore(dim=512, 
                                     session_id=session_id, 
                                     transient=transient)
        holdings.update({"_vecstore": _vecstore})
    vectorize = MockVectorize()
    vecstore = holdings.get("_vecstore")
    assert isinstance(vecstore, VectorStore)
    await vecstore.upsert(bundle, vectorize)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)