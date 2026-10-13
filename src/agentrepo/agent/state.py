"""The agent's LangGraph state: the message history plus a step counter."""

from typing import Annotated
from typing_extensions import NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # add_messages appends new messages instead of overwriting the list. The
    # history holds everything: user question, AI tool requests, tool results, answer.
    messages: Annotated[list[AnyMessage], add_messages]
    # Number of times the agent node has called the LLM (used for the step limit).
    step_count: int
    # Last provider/LLM invocation error handled by an agent node, when any.
    model_error: NotRequired[str]
