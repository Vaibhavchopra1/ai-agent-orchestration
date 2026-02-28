"""
LangGraph StateGraph
--------------------
Defines the agent execution graph:

  [orchestrator] → route → [research | coder | analyst] → [tool_execution] → loop or end
                                                         ↕
                                                    [hitl_gate]
"""

from __future__ import annotations
import os
import json
import asyncio
from typing import TypedDict, Annotated, Literal, AsyncIterator

import anthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.tools.registry import registry
from app.memory.store import SessionStore, RedisCheckpointer


# ── State schema ──────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    session_id: str
    task: str
    messages: Annotated[list, add_messages]
    active_agent: str          # orchestrator | research | coder | analyst
    tool_calls: list[dict]
    tool_results: list[dict]
    hitl_pending: bool
    hitl_approved: bool | None
    final_answer: str | None
    iteration: int


# ── Anthropic client ──────────────────────────────────────────────────────────

_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-6")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

AGENT_SYSTEM_PROMPTS = {
    "orchestrator": (
        "You are an orchestration agent. Decompose the user's task into sub-tasks and decide "
        "which specialist agent (research, coder, analyst) should handle each part. "
        "When the overall task is complete, emit a final_answer."
    ),
    "research": (
        "You are a research agent. Use the web_search tool to gather accurate, up-to-date information. "
        "Summarize your findings clearly."
    ),
    "coder": (
        "You are a coding agent. Write clean, well-commented Python code. "
        "Use read_file/write_file tools to interact with the workspace. "
        "Always test your logic before returning."
    ),
    "analyst": (
        "You are a data analysis agent. Interpret data, identify trends, and produce clear insights. "
        "Use the calculator tool for numerical computations."
    ),
}


# ── Node: run one LLM turn for the active agent ───────────────────────────────

async def agent_node(state: GraphState) -> dict:
    agent = state["active_agent"]
    system = AGENT_SYSTEM_PROMPTS.get(agent, AGENT_SYSTEM_PROMPTS["orchestrator"])
    tools = registry.list_tools()

    response = await _client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        tools=tools,
        messages=state["messages"],
    )

    new_messages = [{"role": "assistant", "content": response.content}]
    tool_calls = []

    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    return {
        "messages": new_messages,
        "tool_calls": tool_calls,
        "iteration": state["iteration"] + 1,
    }


# ── Node: execute tool calls ──────────────────────────────────────────────────

async def tool_node(state: GraphState) -> dict:
    results = []
    tasks = [
        registry.invoke(tc["name"], **tc["input"])
        for tc in state["tool_calls"]
    ]
    outputs = await asyncio.gather(*tasks, return_exceptions=True)

    tool_result_messages = []
    for tc, output in zip(state["tool_calls"], outputs):
        content = str(output) if isinstance(output, Exception) else json.dumps(output)
        results.append({"tool_use_id": tc["id"], "content": content})
        tool_result_messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc["id"], "content": content}],
        })

    return {"messages": tool_result_messages, "tool_results": results, "tool_calls": []}


# ── Node: HITL gate — pause for human approval ────────────────────────────────

async def hitl_node(state: GraphState) -> dict:
    """
    Pauses execution. The API layer writes hitl_approved to state via SessionStore
    and resumes the graph by re-invoking it with the updated state.
    """
    return {"hitl_pending": True}


# ── Node: extract final answer from last assistant message ────────────────────

async def finalize_node(state: GraphState) -> dict:
    for msg in reversed(state["messages"]):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            else:
                text = str(content)
            return {"final_answer": text}
    return {"final_answer": "Task complete."}


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_agent(state: GraphState) -> Literal["tools", "hitl", "finalize", "agent"]:
    if state["iteration"] >= MAX_ITERATIONS:
        return "finalize"
    if state["tool_calls"]:
        # Check if any tool call looks sensitive enough to require HITL
        sensitive = {"write_file"}
        if any(tc["name"] in sensitive for tc in state["tool_calls"]):
            return "hitl"
        return "tools"
    # No tool calls → check if orchestrator signalled completion
    last = state["messages"][-1] if state["messages"] else {}
    content = last.get("content", "")
    text = content if isinstance(content, str) else json.dumps(content)
    if "final_answer" in text.lower() or "task complete" in text.lower():
        return "finalize"
    return "agent"


def route_after_hitl(state: GraphState) -> Literal["tools", "finalize"]:
    if state.get("hitl_approved") is False:
        return "finalize"
    return "tools"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("hitl", hitl_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("agent")

    g.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "hitl": "hitl",
        "finalize": "finalize",
        "agent": "agent",
    })
    g.add_edge("tools", "agent")
    g.add_conditional_edges("hitl", route_after_hitl, {
        "tools": "tools",
        "finalize": "finalize",
    })
    g.add_edge("finalize", END)

    return g.compile()


# ── Runner ────────────────────────────────────────────────────────────────────

compiled_graph = build_graph()


async def run_session(
    session_id: str,
    task: str,
    store: SessionStore,
    checkpointer: RedisCheckpointer,
    resume_state: GraphState | None = None,
) -> AsyncIterator[str]:
    """
    Run or resume an agent session, yielding streaming tokens/events.
    """
    initial_state: GraphState = resume_state or {
        "session_id": session_id,
        "task": task,
        "messages": [{"role": "user", "content": task}],
        "active_agent": "orchestrator",
        "tool_calls": [],
        "tool_results": [],
        "hitl_pending": False,
        "hitl_approved": None,
        "final_answer": None,
        "iteration": 0,
    }

    await store.update(await store.get(session_id) or await store.create(task))

    async for event in compiled_graph.astream(initial_state):
        node_name, node_output = next(iter(event.items()))

        # Checkpoint after every node
        await checkpointer.save(session_id, f"{node_name}_{node_output.get('iteration', 0)}", node_output)

        # Stream a JSON event to the caller
        yield json.dumps({"node": node_name, "data": {
            k: v for k, v in node_output.items()
            if k not in ("messages",)  # don't blast full message history each tick
        }}) + "\n"

        # Persist HITL pause
        if node_name == "hitl":
            last_msg = node_output.get("messages", [{}])[-1] if node_output.get("messages") else {}
            await store.set_hitl(session_id, question="Agent requires approval to write files.", context={})
            break

        if node_output.get("final_answer"):
            session = await store.get(session_id)
            if session:
                session.status = "done"
                session.result = node_output["final_answer"]
                await store.update(session)
