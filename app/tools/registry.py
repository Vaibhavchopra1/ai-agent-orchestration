"""
Tool Registry — agents discover and invoke tools dynamically.
Add a new tool by decorating a function with @registry.register.
"""

from __future__ import annotations
import asyncio
import math
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON-schema style
    fn: Callable[..., Awaitable[Any]]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: dict):
        """Decorator to register an async function as a tool."""
        def decorator(fn: Callable):
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                fn=fn,
            )
            return fn
        return decorator

    async def invoke(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools)}")
        return await self._tools[name].fn(**kwargs)

    def list_tools(self) -> list[dict]:
        """Return tool schemas suitable for passing to the LLM."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self._tools.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ── Singleton registry ────────────────────────────────────────────────────────
registry = ToolRegistry()


# ── Built-in tools ────────────────────────────────────────────────────────────

@registry.register(
    name="web_search",
    description="Search the web for up-to-date information on a topic.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "num_results": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
)
async def web_search(query: str, num_results: int = 5) -> dict:
    """Stub — replace with a real search API (e.g. Tavily, Brave, SerpAPI)."""
    await asyncio.sleep(0.1)  # simulate network
    return {
        "query": query,
        "results": [
            {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i}", "snippet": "..."}
            for i in range(num_results)
        ],
    }


@registry.register(
    name="calculator",
    description="Evaluate a mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "A Python-safe math expression, e.g. '2 ** 10 + sqrt(144)'"},
        },
        "required": ["expression"],
    },
)
async def calculator(expression: str) -> dict:
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["abs"] = abs
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


@registry.register(
    name="read_file",
    description="Read the contents of a text file from the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path inside /workspace"},
        },
        "required": ["path"],
    },
)
async def read_file(path: str) -> dict:
    import aiofiles, os
    safe_path = os.path.normpath(os.path.join("/workspace", path))
    if not safe_path.startswith("/workspace"):
        return {"error": "Path traversal not allowed"}
    try:
        async with aiofiles.open(safe_path) as f:
            content = await f.read()
        return {"path": path, "content": content}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}


@registry.register(
    name="write_file",
    description="Write content to a file in the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path inside /workspace"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    },
)
async def write_file(path: str, content: str) -> dict:
    import aiofiles, os
    safe_path = os.path.normpath(os.path.join("/workspace", path))
    if not safe_path.startswith("/workspace"):
        return {"error": "Path traversal not allowed"}
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    async with aiofiles.open(safe_path, "w") as f:
        await f.write(content)
    return {"path": path, "bytes_written": len(content)}
