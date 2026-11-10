"""Visualize the ARI Deep Agent execution graph with LangGraph."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the repository root is on sys.path
repo_root = Path(__file__).resolve().parent
source_root = repo_root
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from src.agent.config import AgentConfig
from src.agent.deep_agent import build_deep_agent


def get_agent_execution_graph(agent=None):
    """Build or retrieve the execution graph object for ARI's deep agent."""
    if agent is None:
        mock_service = MagicMock()
        mock_config = AgentConfig(api_key="mock_key")
        mock_llm = MagicMock()
        agent = build_deep_agent(mock_service, mock_config, llm=mock_llm)

    return getattr(agent, "_graph", agent).get_graph()


def render_mermaid(agent=None) -> str:
    """Return the compiled graph as Mermaid."""
    return get_agent_execution_graph(agent).draw_mermaid()


def render_ascii(agent=None) -> str:
    """Draw the compiled graph in ASCII when Grandalf can lay it out."""
    return get_agent_execution_graph(agent).draw_ascii()


def export_html(output_path: str = "agent_execution_graph.html", agent=None) -> str:
    """Export an interactive HTML file rendering the graph with Mermaid.js."""
    mermaid_def = render_mermaid(agent)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ARI Deep Agent Execution Graph</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 40px;
            background: #fafafa;
            color: #222;
        }}
        h1 {{ font-size: 24px; margin-bottom: 8px; }}
        p {{ color: #555; margin-bottom: 24px; }}
        .card {{
            background: #ffffff;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            max-width: 900px;
            margin: 0 auto;
        }}
        .mermaid {{ display: flex; justify-content: center; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>ARI Deep Agent Execution Graph</h1>
        <p>Interactive StateGraph topology (LangGraph 5-node architecture)</p>
        <div class="mermaid">
{mermaid_def}
        </div>
    </div>
</body>
</html>
"""
    out = Path(output_path)
    out.write_text(html_content, encoding="utf-8")
    return str(out.resolve())


if __name__ == "__main__":
    print("=" * 60)
    print("ARI Deep Agent Execution Graph (ASCII)")
    print("=" * 60)
    try:
        print(render_ascii())
    except (ImportError, ValueError) as error:
        print(f"ASCII rendering unavailable: {error}", file=sys.stderr)
    print("=" * 60)
    print("ARI Deep Agent Execution Graph (Mermaid Definition)")
    print("=" * 60)
    print(render_mermaid())
    print("\n" + "=" * 60)
    html_file = export_html()
    print(f"Interactive HTML graph exported to: {html_file}")
    print("=" * 60)