"""
The only place that knows which LLM provider is used. The graph just needs
a LangChain chat model that supports bind_tools().
"""

from .config import AgentConfig


def build_llm(config: AgentConfig):
    # Keep provider-specific imports lazy so configuration and fake-model tests
    # do not require the Gemini integration to be installed.
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Bound retries avoid wasting quota when the provider is out of quota.
    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=config.api_key,
        max_retries=1,
        timeout=90,
    )
