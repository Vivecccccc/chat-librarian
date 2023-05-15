from datetime import timedelta, datetime
from fastapi import APIRouter, HTTPException, Request, Response, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from models.api import Settings
from server.constants import (ACCESS_TOKEN_EXPIRE_HOURS,
                              STAGE1_SECRET_KEY,
                              ALGORITHM)

from server.utils import (authenticate_user, 
                          create_access_token, get_user_belongings)
inout_router = APIRouter()

@inout_router.post("/login")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    cookie = sign_new_cookie(user)
    response.set_cookie(key="stage1", value=cookie, max_age=1800)
    return "login successful"

def sign_new_cookie(user):
    access_token_expires_in = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires_in)
    cookie = jwt.encode({"token": access_token, "exp": datetime.utcnow() + timedelta(seconds=1800)}, key=STAGE1_SECRET_KEY, algorithm=ALGORITHM)
    return cookie

def resign_cookie(prev_cookie):
    access_token = jwt.decode(token=prev_cookie, key=STAGE1_SECRET_KEY, algorithms=ALGORITHM)["token"]
    cookie = jwt.encode({"token": access_token, "exp": datetime.utcnow() + timedelta(seconds=1800)}, key=STAGE1_SECRET_KEY, algorithm=ALGORITHM)
    return cookie

@inout_router.get("/logout")
async def logout(request: Request, response: Response):
    if request.cookies:
        response.delete_cookie(key="stage1")
    return "logout successful"

@inout_router.post("/config")
async def configuring(
    request: Request,
    response: Response,
    mode: str,
    chunk_token_length: int,
    embedding_method: str,
    vectorstore: str
):  
    cookies = request.cookies
    if not cookies:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="not authorized or invalid cookies")
    cookie = cookies["stage1"]
    user, belongings = get_user_belongings(request)
    if belongings:
        belongings.clear()
        cookie = resign_cookie(cookie)
    settings = Settings(mode=mode,
                        chunk_size=chunk_token_length,
                        embedding_method=embedding_method,
                        vectorstore=vectorstore)
    belongings.update({"settings": settings})
    response.delete_cookie(key="stage1")
    response.set_cookie(key="stage1", value=cookie, max_age=1800)
    return "configuring successful"