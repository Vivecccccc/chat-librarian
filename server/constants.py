from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

STAGE1_SECRET_KEY = "night_mother"
STAGE2_SECRET_KEY = "what-is-the-meaning-of-life"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN = ".data/reserve/_session/docs/embeddings/"
VECTORSTORE_DOC_SAVE_ROOT_FOR_USER = ".data/transient/_session-%s/docs/embeddings/"
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
AUTH = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    hashed_password: str
    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.dict().values()))

USER_BELONGINGS: Dict[User, Optional[Dict[str, Any]]] = {}
USER_BASIC = {
    "admin": User(username="admin",
                  hashed_password=PWD_CONTEXT.hash("admin")),
    "user1": User(username="user1",
                  hashed_password=PWD_CONTEXT.hash("user1")),
    "user2": User(username="user2",
                  hashed_password=PWD_CONTEXT.hash("user2"))
}