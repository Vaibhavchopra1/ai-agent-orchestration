# AI Agent Orchestration Platform

A production-grade multi-agent orchestration system with autonomous task decomposition, stateful execution via LangGraph, and concurrent session management through a FastAPI gateway.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Gateway                    │
│         (REST API · WebSocket streaming)             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Orchestrator Agent                      │
│     (Task decomposition · Agent routing)             │
└──────┬───────────────┬───────────────┬──────────────┘
       │               │               │
  ┌────▼────┐    ┌─────▼────┐   ┌─────▼────┐
  │Research │    │  Coder   │   │ Analyst  │
  │  Agent  │    │  Agent   │   │  Agent   │
  └────┬────┘    └─────┬────┘   └─────┬────┘
       │               │               │
┌──────▼───────────────▼───────────────▼──────────────┐
│                   Tool Registry                      │
│     web_search · code_exec · file_rw · calculator   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Memory & State Layer (Redis)            │
│     Session state · Conversation history · HITL     │
└─────────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Routing** — Orchestrator decomposes tasks and routes sub-tasks to specialized agents (Research, Coder, Analyst)
- **Stateful Execution** — Full conversation & execution state persisted in Redis via LangGraph checkpointing
- **Tool Registry** — Pluggable tool system; agents discover and invoke tools dynamically
- **Human-in-the-Loop (HITL)** — Agents can pause and request human approval before critical actions
- **Streaming** — Real-time token streaming via WebSocket
- **Concurrent Sessions** — FastAPI handles multiple agent sessions in parallel with async execution

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Claude claude-sonnet-4-6 via Anthropic SDK |
| API | FastAPI + Uvicorn |
| Memory | Redis (session state + checkpointing) |
| Containerization | Docker + Docker Compose |
| Testing | Pytest |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/vaibhav-chopra/ai-agent-orchestration
cd ai-agent-orchestration

# Set environment variables
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Run with Docker
docker-compose up --build

# API is live at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## API Usage

```bash
# Create a new agent session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"task": "Research the latest LangGraph features and write a Python example"}'

# Stream execution (WebSocket)
wscat -c "ws://localhost:8000/sessions/{session_id}/stream"

# Respond to HITL checkpoint
curl -X POST http://localhost:8000/sessions/{session_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "feedback": "Looks good, proceed"}'

# Get session state
curl http://localhost:8000/sessions/{session_id}
```

## Project Structure

```
ai-agent-orchestration/
├── app/
│   ├── main.py               # FastAPI app + routes
│   ├── agents/
│   │   ├── orchestrator.py   # Task decomposition & routing
│   │   ├── research.py       # Web research agent
│   │   ├── coder.py          # Code generation agent
│   │   └── analyst.py        # Data analysis agent
│   ├── tools/
│   │   ├── registry.py       # Tool registry & discovery
│   │   ├── web_search.py     # Web search tool
│   │   ├── code_exec.py      # Sandboxed code execution
│   │   └── calculator.py     # Math/computation tool
│   ├── memory/
│   │   ├── store.py          # Redis session store
│   │   └── checkpointer.py   # LangGraph Redis checkpointer
│   └── graph.py              # LangGraph StateGraph definition
├── tests/
│   ├── test_orchestrator.py
│   ├── test_tools.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```
