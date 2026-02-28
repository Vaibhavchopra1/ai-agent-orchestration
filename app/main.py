"""
FastAPI Gateway
---------------
Exposes the agent orchestration system via REST + WebSocket.

Endpoints:
  POST   /sessions                  — create & start a session
  GET    /sessions/{id}             — get session state
  POST   /sessions/{id}/approve     — resolve a HITL checkpoint
  WS     /sessions/{id}/stream      — real-time execution stream
  GET    /tools                     — list available tools
  GET    /health                    — health check
"""

from __future__ import annotations
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.memory.store import SessionStore, RedisCheckpointer
from app.graph import run_session
from app.tools.registry import registry


# ── App lifecycle ─────────────────────────────────────────────────────────────

store: SessionStore
checkpointer: RedisCheckpointer

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, checkpointer
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    store = SessionStore(redis_url)
    checkpointer = RedisCheckpointer(redis_url)
    yield


app = FastAPI(
    title="AI Agent Orchestration Platform",
    description="Multi-agent system with LangGraph, HITL, and Redis-backed state.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    task: str

class ApproveRequest(BaseModel):
    approved: bool
    feedback: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tools")
async def list_tools():
    return {"tools": registry.list_tools()}


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest):
    session = await store.create(body.task)
    # Kick off execution in the background
    asyncio.create_task(_run_background(session.session_id, body.task))
    return {"session_id": session.session_id, "status": session.status}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = await store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.model_dump()


@app.post("/sessions/{session_id}/approve")
async def approve_hitl(session_id: str, body: ApproveRequest):
    session = await store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status != "waiting_hitl":
        raise HTTPException(400, f"Session is not waiting for approval (status: {session.status})")

    await store.resolve_hitl(session_id, body.approved, body.feedback)

    if body.approved:
        # Resume graph from last checkpoint
        saved = await checkpointer.load_latest(session_id)
        if saved:
            saved["hitl_pending"] = False
            saved["hitl_approved"] = True
            asyncio.create_task(_run_background(session_id, session.task, resume_state=saved))

    return {"status": "resumed" if body.approved else "cancelled"}


@app.websocket("/sessions/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = await store.get(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    try:
        # Stream events as they're produced
        async for event_json in run_session(session_id, session.task, store, checkpointer):
            await websocket.send_text(event_json)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


# ── Background task ───────────────────────────────────────────────────────────

async def _run_background(session_id: str, task: str, resume_state=None):
    session = await store.get(session_id)
    if session:
        session.status = "running"
        await store.update(session)

    try:
        async for _ in run_session(session_id, task, store, checkpointer, resume_state):
            pass  # Events consumed by WebSocket clients; background just drives execution
    except Exception as exc:
        session = await store.get(session_id)
        if session:
            session.status = "error"
            session.result = str(exc)
            await store.update(session)
        raise
