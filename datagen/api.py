"""FastAPI service exposing the DataGen agent over HTTP."""
import json
import os
import secrets
import uuid
from collections import OrderedDict
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from datagen import agent
from datagen.tools import PathNotAllowedError, safe_path

load_dotenv()

MAX_SESSIONS = 500          # oldest sessions are evicted past this
MAX_HISTORY_MESSAGES = 20   # per session, keeps prompts (and cost) bounded

app = FastAPI(
    title="DataGen API",
    description="Generate sample user data from plain-English requests.",
    version="1.0.0",
)

# In-memory chat history keyed by session id. Fine for a single instance;
# a multi-instance deployment would move this to Redis or a database.
sessions: "OrderedDict[str, List[BaseMessage]]" = OrderedDict()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000,
                        examples=["Generate 5 users aged 25-35 with company.com emails"])
    session_id: Optional[str] = Field(None, description="Reuse to continue a conversation.")


class GenerateResponse(BaseModel):
    session_id: str
    reply: str
    data: List[dict]


def require_api_key(x_api_key: Optional[str] = Header(None)):
    """Only callers with DATAGEN_API_KEY can use endpoints that spend OpenAI credits.

    Fails closed: if the server has no key configured, every request is refused.
    """
    expected = os.getenv("DATAGEN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Server has no DATAGEN_API_KEY configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _get_history(session_id: str) -> List[BaseMessage]:
    if session_id in sessions:
        sessions.move_to_end(session_id)
    else:
        sessions[session_id] = []
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    return sessions[session_id]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
def generate(req: GenerateRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = _get_history(session_id)

    reply, data = agent.run_agent(req.prompt, history)

    history.extend([HumanMessage(content=req.prompt), reply])
    del history[:-MAX_HISTORY_MESSAGES]

    return GenerateResponse(session_id=session_id, reply=str(reply.content), data=data)


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
def clear_session(session_id: str):
    if sessions.pop(session_id, None) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"cleared": session_id}


@app.get("/files/{filename}", dependencies=[Depends(require_api_key)])
def get_file(filename: str):
    """Download a JSON file the agent saved."""
    try:
        path = safe_path(filename)
    except PathNotAllowedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return json.loads(path.read_text(encoding="utf-8"))
