"""
Memory & State Layer
--------------------
SessionStore  — stores full agent session data in Redis (TTL-based).
RedisCheckpointer — LangGraph-compatible saver so graphs resume across restarts.
"""

from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel, Field


# ── Data models ───────────────────────────────────────────────────────────────

class HITLCheckpoint(BaseModel):
    question: str
    context: dict = Field(default_factory=dict)
    approved: bool | None = None
    feedback: str | None = None


class AgentSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    status: str = "pending"          # pending | running | waiting_hitl | done | error
    messages: list[dict] = Field(default_factory=list)
    result: str | None = None
    hitl: HITLCheckpoint | None = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Session store ─────────────────────────────────────────────────────────────

class SessionStore:
    TTL = 60 * 60 * 24  # 24 hours

    def __init__(self, redis_url: str):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def create(self, task: str) -> AgentSession:
        session = AgentSession(task=task)
        await self._redis.set(
            self._key(session.session_id),
            session.model_dump_json(),
            ex=self.TTL,
        )
        return session

    async def get(self, session_id: str) -> AgentSession | None:
        raw = await self._redis.get(self._key(session_id))
        if not raw:
            return None
        return AgentSession.model_validate_json(raw)

    async def update(self, session: AgentSession) -> None:
        session.updated_at = datetime.utcnow().isoformat()
        await self._redis.set(
            self._key(session.session_id),
            session.model_dump_json(),
            ex=self.TTL,
        )

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        session = await self.get(session_id)
        if session:
            session.messages.append({"role": role, "content": content})
            await self.update(session)

    async def set_hitl(self, session_id: str, question: str, context: dict) -> None:
        session = await self.get(session_id)
        if session:
            session.status = "waiting_hitl"
            session.hitl = HITLCheckpoint(question=question, context=context)
            await self.update(session)

    async def resolve_hitl(self, session_id: str, approved: bool, feedback: str | None) -> None:
        session = await self.get(session_id)
        if session and session.hitl:
            session.hitl.approved = approved
            session.hitl.feedback = feedback
            session.status = "running"
            await self.update(session)


# ── LangGraph-compatible Redis checkpointer ───────────────────────────────────

class RedisCheckpointer:
    """
    Minimal LangGraph checkpointer backed by Redis.
    Stores graph state snapshots so executions can be resumed after crashes or HITL pauses.
    """

    def __init__(self, redis_url: str):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    def _key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"checkpoint:{thread_id}:{checkpoint_id}"

    def _latest_key(self, thread_id: str) -> str:
        return f"checkpoint:{thread_id}:latest"

    async def save(self, thread_id: str, checkpoint_id: str, state: dict) -> None:
        payload = json.dumps(state)
        await self._redis.set(self._key(thread_id, checkpoint_id), payload, ex=86400)
        await self._redis.set(self._latest_key(thread_id), checkpoint_id, ex=86400)

    async def load_latest(self, thread_id: str) -> dict | None:
        checkpoint_id = await self._redis.get(self._latest_key(thread_id))
        if not checkpoint_id:
            return None
        raw = await self._redis.get(self._key(thread_id, checkpoint_id))
        return json.loads(raw) if raw else None

    async def delete(self, thread_id: str) -> None:
        checkpoint_id = await self._redis.get(self._latest_key(thread_id))
        if checkpoint_id:
            await self._redis.delete(self._key(thread_id, checkpoint_id))
        await self._redis.delete(self._latest_key(thread_id))
