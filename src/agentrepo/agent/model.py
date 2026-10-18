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

    # Keep provider retries bounded. In particular, repeated 429 retries do
    # not help when the account is out of quota and delay the evidence-backed
    # fallback path.
    return ChatGroq(model=config.model, temperature=config.temperature, api_key=config.api_key,
                    max_retries=1, timeout=90)
