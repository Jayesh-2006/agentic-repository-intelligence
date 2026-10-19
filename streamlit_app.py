#!/usr/bin/env python3
"""
Streamlit Web UI for Agentic Repository Intelligence (ARI).

Features:
- Native Windows Explorer folder picker dialog ("📁 Browse...") for selecting local repositories and index folders.
- Automatic index creation when a specified Saved Index Directory (e.g., `data/RAG`) does not exist yet.
- Public GitHub repository importer that stores clones in `<project_root>/data/repos/` and indices in `<project_root>/data/indices/`.
- Subfolder-aware path discovery so this file and `github_importer.py` can live in root or any subdirectory.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import streamlit as st

# Suppress benign ScriptRunContext warnings from LangGraph/LangChain ThreadPoolExecutor workers
for _logger_name in (
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.scriptrunner.script_run_context",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def _find_project_root() -> Path:
    """Locate the project root directory (containing `src/agentrepo` or `main.py`)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (
            (candidate / "src" / "agentrepo").is_dir()
            or (candidate / "main.py").is_file()
            or (candidate / ".env").is_file()
        ):
            return candidate
    return Path.cwd().resolve()


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root()

for _p in (HERE, PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agentrepo import index_repository, save_index  # noqa: E402
from agentrepo.agent.app import (  # noqa: E402
    _STEP_BULLET_RE,
    AppError,
    build_app,
    load_app_config,
    open_repository,
    parse_structured_answer,
    run_query,
)


def _load_github_importer():
    """Import `github_importer` regardless of which folder the user placed it in."""
    for mod_name in ("github_importer", "agentrepo.github_importer", "scripts.github_importer"):
        try:
            return __import__(mod_name, fromlist=["import_github_repo"])
        except ImportError:
            pass
    for candidate_file in PROJECT_ROOT.rglob("github_importer.py"):
        spec = importlib.util.spec_from_file_location("github_importer", candidate_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["github_importer"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Could not locate `github_importer.py` in the project. "
        "Make sure `github_importer.py` is saved in your project folder."
    )


_gh_mod = _load_github_importer()
GitHubImportError = _gh_mod.GitHubImportError
import_github_repo = _gh_mod.import_github_repo
list_cached_repos = _gh_mod.list_cached_repos
ensure_local_index = getattr(_gh_mod, "ensure_local_index", None)
DEFAULT_REPOS_DIR = getattr(_gh_mod, "DEFAULT_REPOS_DIR", PROJECT_ROOT / "data" / "repos")
DEFAULT_INDICES_DIR = getattr(_gh_mod, "DEFAULT_INDICES_DIR", PROJECT_ROOT / "data" / "indices")


EXAMPLE_QUERIES = [
    "What is the main purpose of this repository?",
    "What are the main modules and entry points in this repository?",
    "Walk through the core model or service class and explain its methods.",
    "How are configuration and environment variables handled?",
]


def _pick_folder_dialog(title: str = "Select Folder", initial_dir: str = "") -> str | None:
    """Open a native Windows Explorer / OS folder picker dialog in a dedicated main-thread subprocess."""
    start_dir = initial_dir.strip() if initial_dir and Path(initial_dir.strip()).is_dir() else str(PROJECT_ROOT)
    script = (
        "import sys\n"
        "try:\n"
        "    import tkinter as tk\n"
        "    from tkinter import filedialog\n"
        "    root = tk.Tk()\n"
        "    root.withdraw()\n"
        "    try:\n"
        "        root.wm_attributes('-topmost', 1)\n"
        "    except Exception:\n"
        "        pass\n"
        "    selected = filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2])\n"
        "    root.destroy()\n"
        "    if selected:\n"
        "        sys.stdout.write(selected)\n"
        "except Exception:\n"
        "    pass\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, title, start_dir],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        chosen = (proc.stdout or "").strip()
        if chosen:
            return str(Path(chosen).resolve())
    except Exception:
        pass

    # Windows PowerShell fallback if tkinter is not installed in the Python build
    if os.name == "nt":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$f.Description = '{title}'; "
            "$f.ShowNewFolderButton = $true; "
            "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.SelectedPath }"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            chosen = (proc.stdout or "").strip()
            if chosen:
                return str(Path(chosen).resolve())
        except Exception:
            pass
    return None


def _format_trace_badge(raw: str) -> str:
    """Attach the same semantic icon badges used in RichTerminalUI."""
    line = (raw or "").strip()
    if not line:
        return ""
    if line.startswith("[Main] run started"):
        return f"▶️ `{line}`"
    if "run completed ->" in line:
        if "SUPPORTED" in line:
            return f"✅ **{line}**"
        if "PARTIAL" in line:
            return f"🟡 **{line}**"
        return f"❌ **{line}**"
    if "-> FAILED" in line or "ERROR" in line or "budget NOT met" in line:
        return f"❌ `{line}`"
    if "-> BLOCKED" in line:
        return f"⚡ `{line}`"
    if "retrying with compacted context" in line:
        return f"♻️ `{line}`"
    if "-> EMPTY" in line:
        return f"⚪ `{line}`"
    if "[Main] planning ->" in line or "revise_plan" in line:
        return f"🧭 `{line}`"
    if "[Main] decision" in line or "[Main] model_call(wrap_up)" in line:
        return f"💭 `{line}`"
    if "[Main] evidence_grading" in line:
        return f"⚖️ `{line}`"
    if "[Main] conclusion_validation" in line:
        return f"🛡️ `{line}`"
    if "search_repository" in line:
        return f"🔍 `{line}`"
    if "find_files" in line:
        return f"📁 `{line}`"
    if "find_symbol" in line or "get_ast" in line:
        return f"🧩 `{line}`"
    if any(t in line for t in ("find_callers", "find_callees", "find_importers", "get_dependencies")):
        return f"🔗 `{line}`"
    if "get_file" in line or "get_symbol" in line:
        return f"📄 `{line}`"
    if "write_file" in line or "verify_repository" in line or "run_command" in line:
        return f"🛠️ `{line}`"
    return f"• `{line}`"


def _extract_evidence_rows(sec_body: str) -> list[dict[str, str]]:
    """Parse supporting/reviewed bullet lines into structured table rows."""
    rows: list[dict[str, str]] = []
    for line in (sec_body or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _STEP_BULLET_RE.match(stripped)
        if m:
            gd = m.groupdict()
            step_id = gd.get("step_id") or m.group(1)
            action = gd.get("action") or m.group(2)
            excerpt = gd.get("excerpt") or m.group(3)
            rows.append(
                {
                    "Step": f"[{step_id}]",
                    "Tool & Location": action.strip(),
                    "Recorded Excerpt": excerpt.strip(),
                }
            )
        else:
            rows.append(
                {
                    "Step": "",
                    "Tool & Location": "",
                    "Recorded Excerpt": stripped.lstrip("- "),
                }
            )
    return rows


def _render_structured_response(answer: str, completion: dict | None, trace_lines: list[str]) -> None:
    """Render trace log, evidence status badge, Markdown conclusion, and evidence tables."""
    if trace_lines:
        with st.expander(f"Execution Trace ({len(trace_lines)} events)", expanded=False):
            st.markdown("  \n".join(_format_trace_badge(t) for t in trace_lines))

    support = (completion or {}).get("support", "SUPPORTED")
    c_sup = (completion or {}).get("criteria_supported")
    c_tot = (completion or {}).get("criteria_total")
    ratio = f" ({c_sup}/{c_tot} criteria)" if c_sup is not None and c_tot is not None else ""

    if support == "SUPPORTED":
        st.success(f"**Evidence Status:** `{support}`{ratio}")
    elif support == "PARTIAL":
        st.warning(f"**Evidence Status:** `{support}`{ratio}")
    else:
        st.error(f"**Evidence Status:** `{support}`{ratio}")

    parsed = parse_structured_answer(answer)
    st.markdown(parsed["conclusion"] or "(No conclusion produced)")

    for sec_title, sec_body in parsed["sections"]:
        if sec_title in ("Supporting evidence", "Evidence reviewed but not accepted as support"):
            rows = _extract_evidence_rows(sec_body)
            if rows:
                with st.expander(f"📊 {sec_title} ({len(rows)} item(s))", expanded=True):
                    st.dataframe(rows, width='stretch', hide_index=True)
        elif "unresolved" in sec_title.lower() or "unverified" in sec_title.lower():
            with st.expander(f"⚠️ {sec_title}", expanded=True):
                st.warning(sec_body)
        else:
            with st.expander(f"ℹ️ {sec_title}", expanded=False):
                st.markdown(sec_body)


def _load_active_app(
    repo_path: str,
    index_dir: str | None,
    max_steps: int | None,
    api_key_override: str,
    *,
    auto_create_index: bool = True,
    on_progress=None,
) -> None:
    """Initialize `AgentApp` (building `index_dir` automatically if missing) and store in `st.session_state`."""
    if api_key_override.strip():
        os.environ["GEMINI_API_KEY"] = api_key_override.strip()

    resolved_repo = Path(repo_path.strip()).expanduser()
    if not resolved_repo.is_absolute():
        resolved_repo = (PROJECT_ROOT / resolved_repo).resolve()
    else:
        resolved_repo = resolved_repo.resolve()

    if not resolved_repo.is_dir():
        raise AppError(f"Repository path does not exist or is not a directory: {resolved_repo}")

    def _build_if_missing(target_idx: Path) -> Path:
        has_files = (
            target_idx.is_dir()
            and (target_idx / "symbols.json").is_file()
            and (target_idx / "graph.json").is_file()
            and (target_idx / "chunks.json").is_file()
        )
        if not has_files:
            if on_progress is not None:
                on_progress(f"Building index for {resolved_repo.name} -> {target_idx} ...")
            target_idx.mkdir(parents=True, exist_ok=True)
            idx_obj, _ = index_repository(resolved_repo)
            save_index(idx_obj, target_idx)
            if on_progress is not None:
                on_progress(f"Saved index ({len(idx_obj.symbols)} symbols, {len(idx_obj.chunks)} chunks) to {target_idx}")
        return target_idx

    eff_index: str | None = None
    if index_dir and index_dir.strip():
        raw_idx = Path(index_dir.strip()).expanduser()
        resolved_idx = raw_idx.resolve() if raw_idx.is_absolute() else (PROJECT_ROOT / raw_idx).resolve()
        if auto_create_index:
            resolved_idx = _build_if_missing(resolved_idx)
        eff_index = str(resolved_idx)
    elif auto_create_index:
        resolved_idx = _build_if_missing(DEFAULT_INDICES_DIR / resolved_repo.name)
        eff_index = str(resolved_idx)

    config = load_app_config(max_steps=max_steps)
    service, source = open_repository(str(resolved_repo), eff_index)
    app = build_app(service, config, source)
    st.session_state["ari_app"] = app
    st.session_state["ari_repo_path"] = str(resolved_repo)
    st.session_state["ari_index_dir"] = eff_index


def main() -> None:
    st.set_page_config(
        page_title="ARI — Agentic Repository Intelligence",
        page_icon="🧭",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "ari_app" not in st.session_state:
        st.session_state["ari_app"] = None
    if "local_repo_input" not in st.session_state:
        st.session_state["local_repo_input"] = ""
    if "local_idx_input" not in st.session_state:
        st.session_state["local_idx_input"] = ""

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.title("🧭 ARI Controls")
        st.caption("Agentic Repository Intelligence")

        source_mode = st.radio(
            "Repository Source",
            options=["Public GitHub URL", "Local Folder", "Cached Repos"],
            horizontal=True,
        )

        max_steps = st.slider("Max Agent Steps", min_value=4, max_value=30, value=14, step=1)
        api_key_override = st.text_input(
            "Gemini API Key (optional if in .env)",
            type="password",
            placeholder="Uses GEMINI_API_KEY from .env by default",
        )

        if source_mode == "Public GitHub URL":
            gh_url = st.text_input(
                "GitHub Repository URL or owner/repo",
                placeholder="https://github.com/pallets/click",
            )
            gh_branch = st.text_input("Branch (optional)", placeholder="main")
            force_clone = st.checkbox("Force fresh clone & re-index", value=False)
            st.caption(
                f"Clones to `{DEFAULT_REPOS_DIR.relative_to(PROJECT_ROOT) if DEFAULT_REPOS_DIR.is_relative_to(PROJECT_ROOT) else DEFAULT_REPOS_DIR}` "
                f"and indexes to `{DEFAULT_INDICES_DIR.relative_to(PROJECT_ROOT) if DEFAULT_INDICES_DIR.is_relative_to(PROJECT_ROOT) else DEFAULT_INDICES_DIR}`."
            )
            if st.button("⬇️ Import & Index GitHub Repo", width='stretch', type="primary"):
                if not gh_url.strip():
                    st.error("Please enter a GitHub URL or owner/repo.")
                else:
                    with st.status("Importing repository from GitHub...", expanded=True) as status_box:
                        try:
                            imported = import_github_repo(
                                gh_url.strip(),
                                branch=gh_branch.strip() or None,
                                force_refresh=force_clone,
                                on_progress=status_box.write,
                            )
                            _load_active_app(
                                str(imported.repo_dir),
                                str(imported.index_dir) if imported.index_dir else None,
                                max_steps,
                                api_key_override,
                                auto_create_index=False,
                            )
                            status_box.update(
                                label=f"Loaded {imported.spec.slug} ({imported.symbol_count} symbols)",
                                state="complete",
                                expanded=False,
                            )
                            st.rerun()
                        except (GitHubImportError, AppError, Exception) as exc:
                            status_box.update(label="Import failed", state="error")
                            st.error(str(exc))

        elif source_mode == "Local Folder":
            def _on_browse_repo() -> None:
                picked_dir = _pick_folder_dialog(
                    title="Select Local Repository Folder",
                    initial_dir=st.session_state.get("local_repo_input", ""),
                )
                if picked_dir:
                    st.session_state["local_repo_input"] = picked_dir
                    if not st.session_state.get("local_idx_input"):
                        st.session_state["local_idx_input"] = f"data/indices/{Path(picked_dir).name}"

            def _on_browse_idx() -> None:
                picked_idx = _pick_folder_dialog(
                    title="Select or Create Index Folder",
                    initial_dir=st.session_state.get("local_idx_input", ""),
                )
                if picked_idx:
                    st.session_state["local_idx_input"] = picked_idx

            st.markdown("**Local Repository Path**")
            r_col1, r_col2 = st.columns([3, 1])
            with r_col1:
                local_repo = st.text_input(
                    "Local Repository Path",
                    key="local_repo_input",
                    label_visibility="collapsed",
                    placeholder=r"D:\projects\RAG\data\repos\vqa_system",
                )
            with r_col2:
                st.button(
                    "📁 Browse",
                    key="browse_repo_btn",
                    on_click=_on_browse_repo,
                    width='stretch',
                    help="Open Windows Explorer folder dialog",
                )

            st.markdown("**Saved Index Directory (auto-built if missing)**")
            i_col1, i_col2 = st.columns([3, 1])
            with i_col1:
                local_idx = st.text_input(
                    "Saved Index Directory (optional)",
                    key="local_idx_input",
                    label_visibility="collapsed",
                    placeholder="e.g. data/RAG (auto-created if missing)",
                )
            with i_col2:
                st.button(
                    "📁 Browse",
                    key="browse_idx_btn",
                    on_click=_on_browse_idx,
                    width='stretch',
                    help="Open Windows Explorer folder dialog",
                )

            if st.button("📂 Load & Index Local Repository", width='stretch', type="primary"):
                if not local_repo.strip():
                    st.error("Please select or enter a local repository directory path.")
                else:
                    with st.status("Loading & indexing repository...", expanded=True) as status_box:
                        try:
                            _load_active_app(
                                local_repo,
                                local_idx,
                                max_steps,
                                api_key_override,
                                auto_create_index=True,
                                on_progress=status_box.write,
                            )
                            status_box.update(label="Repository ready!", state="complete", expanded=False)
                            st.rerun()
                        except (AppError, Exception) as exc:
                            status_box.update(label="Failed to load repository", state="error")
                            st.error(str(exc))

        else:  # Cached Repos
            cached = list_cached_repos()
            if not cached:
                st.info("No cached repositories found in `data/repos/` yet.")
            else:
                labels = [name for name, _, _ in cached]
                choice = st.selectbox("Select Cached Repository", options=labels)
                if st.button("🚀 Open Selected Repo", width='stretch', type="primary"):
                    picked = next(item for item in cached if item[0] == choice)
                    with st.status(f"Opening {choice}...", expanded=False):
                        try:
                            _load_active_app(
                                str(picked[1]),
                                str(picked[2]) if picked[2] else None,
                                max_steps,
                                api_key_override,
                                auto_create_index=True,
                            )
                            st.rerun()
                        except (AppError, Exception) as exc:
                            st.error(str(exc))

        # Active Repository Status Card
        app = st.session_state.get("ari_app")
        if app is not None:
            st.divider()
            idx = app.service.index
            retriever_name = getattr(app.service.retriever, "backend_name", type(app.service.retriever).__name__)
            st.subheader("📊 Active Repository")
            st.code(str(idx.root), language=None)
            if st.session_state.get("ari_index_dir"):
                st.caption(f"**Index Dir:** `{st.session_state['ari_index_dir']}`")
            col1, col2 = st.columns(2)
            col1.metric("Symbols", len(idx.symbols))
            col2.metric("Chunks", len(idx.chunks))
            col3, col4 = st.columns(2)
            col3.metric("Python Files", len(idx.files))
            col4.metric("Call/Import Edges", len(idx.edges))
            st.caption(f"**Retriever:** `{retriever_name}`  \n**Model:** `{app.config.model}`")

            b_col1, b_col2 = st.columns(2)
            if b_col1.button("🔄 Refresh Index", width='stretch'):
                res = app.service.refresh()
                if "error" in res:
                    st.error(res["error"])
                else:
                    if st.session_state.get("ari_index_dir") and ensure_local_index is not None:
                        ensure_local_index(idx.root, st.session_state["ari_index_dir"], force_rebuild=True)
                    st.toast(f"Re-indexed {res.get('symbol_count')} symbols!")
                    st.rerun()
            if b_col2.button("🗑️ Clear Chat", width='stretch'):
                st.session_state["messages"] = []
                st.rerun()

    # ---------------- Main Workspace ----------------
    st.header("Agentic Repository Intelligence (ARI)")

    app = st.session_state.get("ari_app")
    if app is None:
        st.info(
            "👈 **Select or import a repository in the sidebar to begin.** "
            "You can paste any public GitHub URL (e.g., `https://github.com/owner/repo`) "
            "or click **📁 Browse** under **Local Folder** to pick a folder from Windows Explorer."
        )
        return

    # Quick-start example query buttons
    st.caption("Quick test queries:")
    q_cols = st.columns(len(EXAMPLE_QUERIES))
    triggered_query: str | None = None
    for i, eq in enumerate(EXAMPLE_QUERIES):
        if q_cols[i].button(eq, key=f"eq_{i}", width='stretch'):
            triggered_query = eq

    # Render conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_structured_response(
                    msg["content"],
                    msg.get("completion"),
                    msg.get("trace", []),
                )

    user_input = st.chat_input("Ask a question about the repository...")
    active_query = triggered_query or user_input

    if active_query:
        st.session_state["messages"].append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            trace_lines: list[str] = []
            with st.status("Investigating repository...", expanded=True) as status_box:
                trace_placeholder = st.empty()
                event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
                result_holder: dict[str, object] = {}

                # Only enqueue strings from worker threads; never call Streamlit UI methods off the main thread
                def on_trace(line: str) -> None:
                    event_queue.put(("trace", line))

                def _worker() -> None:
                    try:
                        ans, comp = run_query(
                            app.agent,
                            active_query,
                            on_line=on_trace,
                            service=app.service,
                        )
                        result_holder["answer"] = ans
                        result_holder["completion"] = comp
                    except Exception as exc:
                        result_holder["error"] = exc
                    finally:
                        event_queue.put(("done", None))

                worker_thread = threading.Thread(target=_worker, daemon=True)
                worker_thread.start()

                while True:
                    kind, payload = event_queue.get()
                    if kind == "trace" and isinstance(payload, str):
                        trace_lines.append(payload)
                        trace_placeholder.markdown(
                            "  \n".join(_format_trace_badge(t) for t in trace_lines)
                        )
                    elif kind == "done":
                        break

                worker_thread.join()

                if "error" in result_holder:
                    status_box.update(label="Investigation failed", state="error", expanded=True)
                    st.error(f"Error: {result_holder['error']}")
                    return

                answer = str(result_holder.get("answer") or "")
                completion = result_holder.get("completion")  # type: ignore[assignment]
                support = (completion or {}).get("support", "SUPPORTED") if isinstance(completion, dict) else "SUPPORTED"
                status_box.update(
                    label=f"Investigation complete — {support}",
                    state="complete",
                    expanded=False,
                )

            _render_structured_response(answer, completion, trace_lines) # type: ignore
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "completion": completion,
                    "trace": trace_lines,
                }
            )


if __name__ == "__main__":
    main()