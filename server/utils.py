from datetime import timedelta, datetime
from typing import Dict, Optional
from fastapi import HTTPException, Request, status
from jose import ExpiredSignatureError, JWTError, jwt
from constants import *

def authenticate_user(username: str, password: str) -> Optional[User]:
    user = USER_BASIC.get(username, None)
    if user and PWD_CONTEXT.verify(password, user.hashed_password):
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