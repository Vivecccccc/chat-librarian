import os
import uvicorn
from fastapi import FastAPI, File, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

from models.api import *

app = FastAPI()
bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    files: List[UploadFile] = File(...),
    token: HTTPAuthorizationCredentials = Depends(validate_token),
):
    pass