import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
from server.router.file import file_router
from server.router.inout import inout_router
from server.router.chat import conversation_router

from fastapi import FastAPI
from fastapi import FastAPI

app = FastAPI()
app.include_router(file_router)
app.include_router(inout_router)
app.include_router(conversation_router)

# @app.get("/echo")
# async def echo(request: Request, response: Response):
#     print(request.cookies)
#     holdings = get_user_belongings(request)
#     holdings.update({"yes": "2"})
#     USER_BELONGINGS.update({get_current_user_from_cookies(request.cookies): {"what": "1"}})
#     return "echo"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)