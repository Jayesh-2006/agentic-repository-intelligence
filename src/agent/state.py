"""Agent state schema and shared LangChain and LangGraph imports."""

from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import NotRequired, TypedDict

__all__ = [
    "AIMessage",
    "AgentState",
    "AnyMessage",
    "END",
    "HumanMessage",
    "START",
    "StateGraph",
    "StructuredTool",
    "SystemMessage",
    "ToolMessage",
    "ToolNode",
    "add_messages",
]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int
    model_error: NotRequired[str]