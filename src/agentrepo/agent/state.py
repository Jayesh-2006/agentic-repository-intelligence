"""LangGraph state definitions and message/tool primitives."""

from __future__ import annotations

import copy
from typing import Annotated, Any, Callable
import uuid

from pydantic import BaseModel, ValidationError
from typing_extensions import NotRequired, TypedDict

try:
    from langchain_core.messages import (  # type: ignore[import-not-found]
        AIMessage,
        AnyMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.tools import StructuredTool  # type: ignore[import-not-found]
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
    from langgraph.graph.message import add_messages  # type: ignore[import-not-found]
    from langgraph.prebuilt import ToolNode  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - used when langchain/langgraph are not installed
    START = "__start__"
    END = "__end__"

    class _BaseMessage:
        type: str = "base"

        def __init__(
            self,
            content: Any = "",
            *,
            id: str | None = None,
            name: str | None = None,
            additional_kwargs: dict[str, Any] | None = None,
            response_metadata: dict[str, Any] | None = None,
            usage_metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self.content = content
            self.id = id or f"msg_{uuid.uuid4().hex[:12]}"
            self.name = name
            self.additional_kwargs = dict(additional_kwargs or {})
            self.response_metadata = dict(response_metadata or {})
            self.usage_metadata = dict(usage_metadata or {})
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_copy(self, *, update: dict[str, Any] | None = None) -> "_BaseMessage":
            clone = copy.copy(self)
            for k, v in (update or {}).items():
                setattr(clone, k, v)
            return clone

    class HumanMessage(_BaseMessage):
        type = "human"

    class SystemMessage(_BaseMessage):
        type = "system"

    class AIMessage(_BaseMessage):
        type = "ai"

        def __init__(
            self,
            content: Any = "",
            *,
            tool_calls: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(content, **kwargs)
            self.tool_calls = list(tool_calls or [])

    class ToolMessage(_BaseMessage):
        type = "tool"

        def __init__(
            self,
            content: Any = "",
            *,
            tool_call_id: str = "",
            name: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(content, name=name, **kwargs)
            self.tool_call_id = tool_call_id

    AnyMessage = _BaseMessage

    def _coerce_message(item: Any) -> Any:
        if isinstance(item, _BaseMessage):
            return item
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("type") or "human").lower()
            content = item.get("content", "")
            if role in ("user", "human"):
                return HumanMessage(content=content)
            if role in ("assistant", "ai"):
                return AIMessage(content=content, tool_calls=item.get("tool_calls"))
            if role == "system":
                return SystemMessage(content=content)
            if role == "tool":
                return ToolMessage(
                    content=content,
                    tool_call_id=str(item.get("tool_call_id") or ""),
                    name=item.get("name"),
                )
        return item

    def add_messages(left: list | None, right: list | Any) -> list:
        base = [_coerce_message(m) for m in (left or [])]
        incoming = right if isinstance(right, list) else [right]
        for m in incoming:
            if m is not None:
                base.append(_coerce_message(m))
        return base

    class StructuredTool:
        def __init__(
            self,
            *,
            func: Callable[..., Any],
            name: str,
            description: str,
            args_schema: type[BaseModel] | None = None,
            handle_validation_error: Callable[[ValidationError], str] | bool = False,
        ) -> None:
            self.func = func
            self.name = name
            self.description = description
            self.args_schema = args_schema
            self.handle_validation_error = handle_validation_error

        @classmethod
        def from_function(
            cls,
            func: Callable[..., Any],
            name: str | None = None,
            description: str | None = None,
            args_schema: type[BaseModel] | None = None,
            handle_validation_error: Callable[[ValidationError], str] | bool = False,
            **_kwargs: Any,
        ) -> "StructuredTool":
            return cls(
                func=func,
                name=name or getattr(func, "__name__", "tool"),
                description=description or (getattr(func, "__doc__", "") or "").strip(),
                args_schema=args_schema,
                handle_validation_error=handle_validation_error,
            )

        def invoke(self, tool_input: dict[str, Any] | Any, **_kwargs: Any) -> Any:
            args = dict(tool_input) if isinstance(tool_input, dict) else {}
            if self.args_schema is not None:
                try:
                    validated = self.args_schema(**args)
                    fields_map = getattr(type(validated), "model_fields", None) or getattr(type(validated), "__fields__", {})
                    args = {k: getattr(validated, k) for k in fields_map}
                except ValidationError as exc:
                    if callable(self.handle_validation_error):
                        return self.handle_validation_error(exc)
                    raise
            return self.func(**args)

    class ToolNode:
        def __init__(self, tools: list[StructuredTool], *, handle_tool_errors: bool = False) -> None:
            self.tools_by_name = {t.name: t for t in tools}
            self.handle_tool_errors = handle_tool_errors

        def invoke(self, state: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            messages = state.get("messages") or []
            if not messages:
                return {"messages": []}
            last = messages[-1]
            tool_calls = (
                last.get("tool_calls")
                if isinstance(last, dict)
                else getattr(last, "tool_calls", None)
            ) or []
            out_messages: list[ToolMessage] = []
            for idx, call in enumerate(tool_calls):
                t_name = str(call.get("name") or "")
                t_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                t_id = str(call.get("id") or f"call_{idx + 1}")
                tool = self.tools_by_name.get(t_name)
                if tool is None:
                    content = f'{{"error": {{"code": "unknown_tool", "message": "unknown tool {t_name!r}"}}}}'
                else:
                    try:
                        res = tool.invoke(t_args)
                        content = str(res)
                    except Exception as exc:
                        if self.handle_tool_errors:
                            content = f"Error: {exc}"
                        else:
                            raise
                out_messages.append(ToolMessage(content=content, name=t_name, tool_call_id=t_id))
            return {"messages": out_messages}

    class _CompiledStateGraph:
        def __init__(
            self,
            nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
            edges: dict[str, str],
            conditional_edges: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]],
            recursion_limit: int = 50,
        ) -> None:
            self._nodes = nodes
            self._edges = edges
            self._conditional_edges = conditional_edges
            self._recursion_limit = recursion_limit

        def with_config(self, config: dict[str, Any] | None = None, **kwargs: Any) -> "_CompiledStateGraph":
            merged = {**(config or {}), **kwargs}
            limit = int(merged.get("recursion_limit", self._recursion_limit))
            return _CompiledStateGraph(
                self._nodes, self._edges, self._conditional_edges, recursion_limit=limit
            )

        def stream(self, input_state: dict[str, Any], stream_mode: str = "updates", **_kwargs: Any):
            state: dict[str, Any] = dict(input_state or {})
            state["messages"] = add_messages([], state.get("messages") or [])
            current = self._edges.get(START)
            steps = 0
            while current and current != END:
                steps += 1
                if steps > self._recursion_limit:
                    raise RuntimeError(f"Recursion limit of {self._recursion_limit} reached")
                node_fn = self._nodes[current]
                update = node_fn(dict(state)) or {}
                if "messages" in update:
                    state["messages"] = add_messages(state.get("messages"), update["messages"])
                for k, v in update.items():
                    if k != "messages":
                        state[k] = v
                if stream_mode == "updates":
                    yield {current: update}
                else:
                    yield dict(state)
                if current in self._conditional_edges:
                    router, mapping = self._conditional_edges[current]
                    route_key = router(state)
                    current = mapping.get(route_key, route_key)
                else:
                    current = self._edges.get(current, END)

        def invoke(self, input_state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            state: dict[str, Any] = dict(input_state or {})
            state["messages"] = add_messages([], state.get("messages") or [])
            for chunk in self.stream(state, stream_mode="updates", **kwargs):
                for _node, update in chunk.items():
                    if "messages" in update:
                        state["messages"] = add_messages(state.get("messages"), update["messages"])
                    for k, v in update.items():
                        if k != "messages":
                            state[k] = v
            return state

    class StateGraph:
        def __init__(self, state_schema: Any) -> None:
            self.state_schema = state_schema
            self._nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
            self._edges: dict[str, str] = {}
            self._conditional_edges: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]] = {}

        def add_node(self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
            self._nodes[name] = fn

        def add_edge(self, start_key: str, end_key: str) -> None:
            self._edges[start_key] = end_key

        def add_conditional_edges(
            self,
            source: str,
            path: Callable[[dict[str, Any]], str],
            path_map: dict[str, str],
        ) -> None:
            self._conditional_edges[source] = (path, dict(path_map))

        def compile(self) -> _CompiledStateGraph:
            return _CompiledStateGraph(self._nodes, self._edges, self._conditional_edges)


class AgentState(TypedDict):
    # add_messages appends new messages instead of overwriting the list. The
    # history holds everything: user question, AI tool requests, tool results, answer.
    messages: Annotated[list[AnyMessage], add_messages]
    # Number of times the agent node has called the LLM (used for the step limit).
    step_count: int
    # Last provider/LLM invocation error handled by an agent node, when any.
    model_error: NotRequired[str]
