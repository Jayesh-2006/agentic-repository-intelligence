"""
The only place that knows which LLM provider is used. To switch providers,
change build_llm(); the graph just needs a LangChain chat model that supports
bind_tools().
"""

from .config import AgentConfig


def build_llm(config: AgentConfig):
    # Imported here so everything else (config, tests with a fake model) works
    # even if langchain-groq is not installed.
    from langchain_groq import ChatGroq

    # max_retries: one transient 429/5xx must not end the whole run.
    return ChatGroq(model=config.model, temperature=config.temperature, api_key=config.api_key,
                    max_retries=4, timeout=90)
