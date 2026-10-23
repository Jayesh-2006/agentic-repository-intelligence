#!/usr/bin/env python3
"""Streamlit Web UI for Agentic Repository Intelligence (ARI)."""

from __future__ import annotations

import html
import importlib.util
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]

USER_AVATAR = "assets/user_avatar.jpg"

for _logger_name in (
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.scriptrunner.script_run_context",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def _find_project_root() -> Path:
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
from agentrepo.ui_explorer import (  # noqa: E402
    render_repository_explorer as _ui_render_repository_explorer,
)
from agentrepo.agent.app import (  # noqa: E402
    _STEP_BULLET_RE,
    AppError,
    build_app,
    load_app_config,
    open_repository,
    parse_structured_answer,
    run_query,
)


def render_repository_explorer(st_mod, app) -> None:
    _ui_render_repository_explorer(st_mod, app)


def _load_github_importer():
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
    raise ImportError("Could not locate `github_importer.py` in the project.")


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
            rows.append({
                "Step": f"[{step_id}]",
                "Tool & Location": action.strip(),
                "Recorded Excerpt": excerpt.strip(),
            })
        else:
            rows.append({
                "Step": "",
                "Tool & Location": "",
                "Recorded Excerpt": stripped.lstrip("- "),
            })
    return rows


_CRITERION_STATUS_BADGES = {
    "SUPPORTED": ("✅ PASS (SUPPORTED)", "success"),
    "PARTIALLY_SUPPORTED": ("🟡 PARTIAL", "warning"),
    "PARTIAL": ("🟡 PARTIAL", "warning"),
    "UNSUPPORTED": ("❌ FAIL (UNSUPPORTED)", "error"),
    "GRADING_FAILED": ("⚠️ GRADING FAILED", "error"),
    "UNKNOWN": ("❔ UNKNOWN / UNGRADED", "info"),
    "REMOVED": ("➖ REMOVED", "info"),
}


def _render_runtime_telemetry(completion: dict | None, st_mod=None) -> None:
    ui = st_mod or st
    if not isinstance(completion, dict):
        return
    telemetry = completion.get("telemetry")
    if not isinstance(telemetry, dict):
        return

    total_ms = int(telemetry.get("total_ms", 0))
    llm_calls = int(telemetry.get("llm_calls", 0))
    tool_calls = int(telemetry.get("tool_calls", 0))
    total_tokens = int(telemetry.get("total_tokens", 0))
    in_tokens = int(telemetry.get("input_tokens", 0))
    out_tokens = int(telemetry.get("output_tokens", 0))
    cache_hits = int(telemetry.get("cache_hits", 0))
    dup_calls = int(telemetry.get("duplicate_tool_calls", 0))
    rep_blocks = int(telemetry.get("repeated_action_blocks", 0))

    sec_str = f"{total_ms / 1000.0:.2f}s" if total_ms >= 1000 else f"{total_ms}ms"
    tok_str = f"{total_tokens:,} tok ({in_tokens:,} in / {out_tokens:,} out)" if telemetry.get("token_data") else "tokens n/a"

    pills = [
        f"⏱️ **{sec_str}**",
        f"🧠 **{llm_calls}** LLM call{'s' if llm_calls != 1 else ''}",
        f"🔧 **{tool_calls}** tool call{'s' if tool_calls != 1 else ''}",
        f"🪙 **{tok_str}**",
    ]
    if cache_hits:
        pills.append(f"⚡ **{cache_hits}** grader cache hit{'s' if cache_hits != 1 else ''}")
    if rep_blocks:
        pills.append(f"🛡️ **{rep_blocks}** redundant read{'s' if rep_blocks != 1 else ''} blocked")

    ui.caption("  ·  ".join(pills)) # pyright: ignore[reportOptionalMemberAccess]

    with ui.expander(f"⚡ Runtime Telemetry & Stage Breakdown ({sec_str} · {llm_calls} LLM · {tool_calls} tools)", expanded=False):
        if hasattr(ui, "columns"):
            m1, m2, m3, m4 = ui.columns(4)
            m1.metric("Wall Latency", sec_str)
            m2.metric("LLM Calls", llm_calls)
            m3.metric("Tool Executions", f"{tool_calls} ({telemetry.get('tool_ms', 0)} ms)")
            m4.metric("Total Tokens", f"{total_tokens:,}" if telemetry.get("token_data") else "N/A")

        stages = telemetry.get("stages") or {}
        if stages:
            stage_rows = [
                {
                    "Agent Stage": stg_name,
                    "LLM Calls": int(stg_data.get("calls", 0)),
                    "Input Tokens": int(stg_data.get("input_tokens", 0)),
                    "Output Tokens": int(stg_data.get("output_tokens", 0)),
                    "Latency (ms)": int(stg_data.get("latency_ms", 0)),
                }
                for stg_name, stg_data in sorted(stages.items())
                if isinstance(stg_data, dict)
            ]
            if stage_rows:
                ui.markdown("##### 🧠 Per-Stage LLM Breakdown")
                ui.dataframe(stage_rows, hide_index=True, width="stretch")

        tool_counts = telemetry.get("tool_counts") or {}
        repo_metrics = telemetry.get("repository_metrics") or {}
        if tool_counts or repo_metrics:
            tool_rows = []
            for t_name, cnt in sorted(tool_counts.items()):
                r_info = repo_metrics.get(t_name) or {}
                tool_rows.append({
                    "Tool / Service Operation": t_name,
                    "Invocations": cnt,
                    "Service Latency (ms)": int(r_info.get("latency_ms", 0)),
                })
            for r_name, r_info in sorted(repo_metrics.items()):
                if r_name not in tool_counts:
                    tool_rows.append({
                        "Tool / Service Operation": f"service.{r_name}",
                        "Invocations": int(r_info.get("calls", 0)),
                        "Service Latency (ms)": int(r_info.get("latency_ms", 0)),
                    })
            if tool_rows:
                ui.markdown("##### 🔧 Tool & Repository Operation Timings")
                ui.dataframe(tool_rows, hide_index=True, width="stretch")

        if dup_calls or rep_blocks or telemetry.get("deterministic_followups"):
            ui.caption(
                f"**Guard & Control Telemetry:** Duplicate calls: `{dup_calls}` · "
                f"Repeated-read blocks: `{rep_blocks}` · "
                f"Auto follow-ups: `{len(telemetry.get('deterministic_followups') or [])}`"
            )


def _format_colored_diff_html(unified_diff: str) -> str:
    if not unified_diff.strip():
        return '<div class="ari-diff-box"><i>(No line changes)</i></div>'
    lines_html: list[str] = []
    for raw_line in unified_diff.splitlines():
        esc = html.escape(raw_line)
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            lines_html.append(f'<div class="ari-diff-hdr">{esc}</div>')
        elif raw_line.startswith("@@"):
            lines_html.append(f'<div class="ari-diff-hunk">{esc}</div>')
        elif raw_line.startswith("+"):
            lines_html.append(f'<div class="ari-diff-add">{esc}</div>')
        elif raw_line.startswith("-"):
            lines_html.append(f'<div class="ari-diff-del">{esc}</div>')
        else:
            lines_html.append(f'<div class="ari-diff-ctx">{esc}</div>')
    body = "".join(lines_html)
    return f"""
    <style>
      .ari-diff-box {{
        background: #0d131f;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        max-height: 420px;
        overflow-y: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.8rem;
        line-height: 1.5;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }}
      .ari-diff-hdr {{ color: #94a3b8; font-weight: 700; }}
      .ari-diff-hunk {{ color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 1px 6px; border-radius: 4px; margin: 2px 0; }}
      .ari-diff-add {{ color: #4ade80; background: rgba(34, 197, 94, 0.14); padding: 0 6px; border-left: 3px solid #22c55e; }}
      .ari-diff-del {{ color: #f87171; background: rgba(239, 68, 68, 0.14); padding: 0 6px; border-left: 3px solid #ef4444; }}
      .ari-diff-ctx {{ color: #cbd5e1; padding: 0 6px; }}
    </style>
    <div class="ari-diff-box">{body}</div>
    """


def _render_mutation_diff_card(completion: dict | None, msg_key: str = "live", st_mod=None) -> None:
    ui = st_mod or st
    if not isinstance(completion, dict):
        return
    mutation = completion.get("mutation")
    if not isinstance(mutation, dict):
        return

    file_diffs = [d for d in (mutation.get("file_diffs") or []) if isinstance(d, dict)]
    changed_files = list(mutation.get("changed_files") or [d.get("file") for d in file_diffs if d.get("file")])
    if not file_diffs and not changed_files:
        return

    tx_id = str(mutation.get("tx_id") or msg_key)
    added_total = int(mutation.get("total_added_lines", 0))
    removed_total = int(mutation.get("total_removed_lines", 0))
    auto_rolled_back = bool(mutation.get("rolled_back"))
    user_decision = mutation.get("user_decision")

    if user_decision == "rolled_back":
        status_banner = "⏪ **ROLLED BACK BY USER** — Pre-task checkpoint restored and index refreshed."
        banner_kind = "warning"
    elif user_decision == "approved":
        status_banner = "✅ **CHANGES KEPT & APPROVED** — Checkpoint finalized."
        banner_kind = "success"
    elif auto_rolled_back:
        status_banner = "🛡️ **AUTO-ROLLED BACK BY SAFETY ENGINE** — Verification failed; repository restored to pre-task checkpoint."
        banner_kind = "error"
    else:
        status_banner = f"🛠️ **GUARDED MUTATION APPLIED** — {len(changed_files)} file(s) modified (`+{added_total}` / `-{removed_total}` lines)."
        banner_kind = "info"

    with ui.expander(
        f"🛡️ Guarded Mutation Diff & Checkpoint Gate ({len(changed_files)} file(s) · +{added_total} / -{removed_total} lines)",
        expanded=True,
    ):
        if banner_kind == "success":
            ui.success(status_banner)
        elif banner_kind == "warning":
            ui.warning(status_banner)
        elif banner_kind == "error":
            ui.error(status_banner)
        else:
            ui.info(status_banner)

        ui.caption(
            f"**Transaction:** `{tx_id}` · "
            f"**Checkpoint:** `{mutation.get('checkpoint_id') or 'snapshot'}` · "
            f"**Verification State:** `{mutation.get('state') or mutation.get('status')}` · "
            f"**Repair Attempts:** `{mutation.get('repair_attempts', 0)}`"
        )

        app = st.session_state.get("ari_app") if st is not None and hasattr(st, "session_state") else None
        if not auto_rolled_back and user_decision not in ("approved", "rolled_back") and app is not None and hasattr(ui, "columns"):
            b1, b2 = ui.columns(2)
            if b1.button("✅ Keep Changes", key=f"keep_mut_{tx_id}_{msg_key}", width="stretch", type="primary"):
                app.service.approve_mutation_checkpoint(mutation)
                mutation["user_decision"] = "approved"
                mutation["can_rollback"] = False
                if hasattr(st, "toast"):
                    st.toast("Changes approved and checkpoint finalized!")
                if hasattr(st, "rerun"):
                    st.rerun()
            if b2.button("⏪ Rollback Checkpoint", key=f"rb_mut_{tx_id}_{msg_key}", width="stretch"):
                rb_res = app.service.apply_mutation_rollback(mutation)
                if rb_res.get("error"):
                    ui.error(f"Rollback failed: {rb_res['error']}")
                else:
                    mutation["user_decision"] = "rolled_back"
                    mutation["rolled_back"] = True
                    mutation["can_rollback"] = False
                    if hasattr(st, "toast"):
                        st.toast("Rolled back changes to pre-task checkpoint and refreshed index!")
                    if hasattr(st, "rerun"):
                        st.rerun()

        for f_idx, fdiff in enumerate(file_diffs):
            f_path = str(fdiff.get("file") or f"file_{f_idx}")
            chg_type = str(fdiff.get("change_type") or "modified").upper()
            f_add = int(fdiff.get("added_lines", 0))
            f_rem = int(fdiff.get("removed_lines", 0))
            ui.markdown(f"**📄 `{f_path}`** (`{chg_type}` · `+{f_add}` / `-{f_rem}` lines)")

            if hasattr(ui, "tabs"):
                u_tab, sbs_tab = ui.tabs(["Unified Git Diff", "Side-by-Side Comparison"])
                with u_tab:
                    diff_html = _format_colored_diff_html(str(fdiff.get("unified_diff") or ""))
                    if hasattr(ui, "html"):
                        ui.html(diff_html)
                    else:
                        ui.markdown(diff_html, unsafe_allow_html=True)
                with sbs_tab:
                    col_before, col_after = ui.columns(2)
                    lang = Path(f_path).suffix.lstrip(".") or "python"
                    with col_before:
                        ui.caption("Before (Checkpoint Snapshot)")
                        ui.code(str(fdiff.get("before_content") or "(new file — did not exist)"), language=lang, line_numbers=True)
                    with col_after:
                        ui.caption("After (Agent Write)")
                        ui.code(str(fdiff.get("after_content") or "(file deleted)"), language=lang, line_numbers=True)
            else:
                ui.code(str(fdiff.get("unified_diff") or ""), language="diff")


def _render_acceptance_criteria(completion: dict | None, st_mod=None) -> None:
    ui = st_mod or st
    if not isinstance(completion, dict):
        return

    raw_criteria = completion.get("criteria")
    raw_initial = completion.get("initial_criteria")
    removed_criteria = completion.get("removed_criteria") or []
    summary = completion.get("criteria_summary") if isinstance(completion.get("criteria_summary"), dict) else {}

    criteria_list = [c for c in (raw_criteria or []) if isinstance(c, dict)]
    initial_list = [c for c in (raw_initial or []) if isinstance(c, dict)]

    c_sup = completion.get("criteria_supported")
    c_tot = completion.get("criteria_total")
    if c_sup is None:
        c_sup = summary.get("supported", sum(1 for c in criteria_list if c.get("satisfied")))
    if c_tot is None:
        c_tot = summary.get("total", len(criteria_list))

    if not criteria_list and not initial_list and c_tot is None:
        return

    any_revised = bool(summary.get("any_revised")) or any(c.get("revised") for c in criteria_list) or bool(removed_criteria)
    rev_tag = " · 🔄 revised" if any_revised else ""
    header_ratio = f"{c_sup}/{c_tot} passed" if (c_sup is not None and c_tot is not None) else "status unknown"

    with ui.expander(f"🎯 Acceptance Criteria ({header_ratio}{rev_tag})", expanded=True):
        source_label = summary.get("source_label") or summary.get("source") or "Standard"
        part_cnt = summary.get("partially_supported", 0)
        unsup_cnt = summary.get("unsupported", 0)
        unk_cnt = summary.get("unknown", 0) + summary.get("grading_failed", 0)
        init_tot = summary.get("initial_total", len(initial_list) or c_tot or 0)

        summary_bits = [
            f"**Overall:** `{c_sup}/{c_tot}` passed",
            f"**Initial Committed:** `{init_tot}`",
            f"**Final Evaluated:** `{c_tot}`",
            f"**Source:** `{source_label}`",
        ]
        if part_cnt:
            summary_bits.append(f"**Partial:** `{part_cnt}`")
        if unsup_cnt:
            summary_bits.append(f"**Failed / Unsupported:** `{unsup_cnt}`")
        if unk_cnt:
            summary_bits.append(f"**Unknown / Grading Failed:** `{unk_cnt}`")
        ui.caption("  ·  ".join(summary_bits))

        if summary.get("planning_error"):
            ui.warning(f"**Planning fallback used:** {summary['planning_error']}")
        if summary.get("grading_failed_flag") or completion.get("support") == "GRADING_FAILED":
            ui.error("Evidence grading failed or could not complete; criteria could not be verified as satisfied.")
        if completion.get("validation_status") == "validation_unavailable: rate_limit":
            ui.warning("Final claim validation was rate-limited; criteria evidence was recorded but final prose claims remain unverified.")

        if criteria_list:
            overview_rows = []
            for c in criteria_list:
                cid = str(c.get("id") or "?")
                passed = bool(c.get("satisfied"))
                status_raw = str(c.get("status") or ("SUPPORTED" if passed else "UNSUPPORTED")).upper()
                badge_label, _ = _CRITERION_STATUS_BADGES.get(status_raw, (status_raw, "info"))
                ev_ids = [str(x) for x in (c.get("evidence_ids") or []) if isinstance(x, str)]
                overview_rows.append({
                    "ID": f"[{cid}]",
                    "Result": badge_label,
                    "Acceptance Criterion": str(c.get("description") or ""),
                    "Evidence Steps": ", ".join(f"[{sid}]" for sid in ev_ids) if ev_ids else "None",
                })
            ui.dataframe(overview_rows, hide_index=True, width="stretch")

        if initial_list and (any_revised or len(initial_list) != len(criteria_list)):
            ui.markdown("##### 🔄 Initial vs. Final Criteria Comparison")
            comp_rows = []
            final_by_id = {str(c.get("id")): c for c in criteria_list}
            seen_ids = set()
            for ic in initial_list:
                cid = str(ic.get("id", "?"))
                seen_ids.add(cid)
                fc = final_by_id.get(cid)
                if fc is None:
                    comp_rows.append({
                        "ID": f"[{cid}]",
                        "Change": "➖ REMOVED",
                        "Initial Criterion (Start of Query)": str(ic.get("description", "")),
                        "Final Evaluated Criterion": "(removed / superseded)",
                        "Final Status": "REMOVED",
                    })
                else:
                    chg = str(fc.get("change_type") or ("revised" if fc.get("revised") else "unchanged")).upper()
                    comp_rows.append({
                        "ID": f"[{cid}]",
                        "Change": f"🔄 {chg}" if fc.get("revised") else "✓ UNCHANGED",
                        "Initial Criterion (Start of Query)": str(ic.get("description", "")),
                        "Final Evaluated Criterion": str(fc.get("description", "")),
                        "Final Status": str(fc.get("status") or ("SUPPORTED" if fc.get("satisfied") else "UNSUPPORTED")),
                    })
            for fc in criteria_list:
                cid = str(fc.get("id", "?"))
                if cid not in seen_ids:
                    comp_rows.append({
                        "ID": f"[{cid}]",
                        "Change": "➕ ADDED",
                        "Initial Criterion (Start of Query)": "(not in initial plan)",
                        "Final Evaluated Criterion": str(fc.get("description", "")),
                        "Final Status": str(fc.get("status") or ("SUPPORTED" if fc.get("satisfied") else "UNSUPPORTED")),
                    })
            if comp_rows:
                ui.dataframe(comp_rows, hide_index=True, width="stretch")

        if not criteria_list and not removed_criteria:
            ui.info("Detailed per-criterion records were not available for this response.")
            return

        for c in [*criteria_list, *[r for r in removed_criteria if isinstance(r, dict)]]:
            cid = str(c.get("id") or "?")
            status_raw = str(
                c.get("status")
                or ("SUPPORTED" if c.get("satisfied") else "UNSUPPORTED")
            ).upper()
            badge_label, _ = _CRITERION_STATUS_BADGES.get(status_raw, (status_raw, "info"))
            rev_label = ""
            if c.get("revised"):
                ctype = str(c.get("change_type") or "revised").upper()
                rev_label = f"  ·  `🔄 {ctype}`"

            desc = str(c.get("description") or "(no description)")
            init_desc = c.get("initial_description")
            needed = str(c.get("evidence_needed") or "").strip()
            explanation = str(c.get("explanation") or c.get("note") or "").strip()

            ui.markdown(f"**`[{cid}]` {badge_label}{rev_label}** — {desc}")
            if c.get("revised") and isinstance(init_desc, str) and init_desc and init_desc != desc:
                ui.caption(f"**Initial criterion at query start:** {init_desc}")
            if c.get("revision_note"):
                ui.caption(f"**Revision note:** {c['revision_note']}")
            if needed:
                ui.caption(f"**Expected evidence:** {needed}")
            if explanation:
                ui.caption(f"**Assessment explanation:** {explanation}")

            ev_refs = [r for r in (c.get("evidence_refs") or []) if isinstance(r, dict)]
            ev_ids = [str(x) for x in (c.get("evidence_ids") or []) if isinstance(x, str)]
            if ev_refs:
                ref_rows = [
                    {
                        "Step": f"[{ref.get('step_id', '?')}]",
                        "Tool & Action": ref.get("action") or ref.get("tool") or "",
                        "Location": ref.get("location") or "",
                        "Citation Check": "Accepted" if ref.get("accepted", True) else "Rejected by citation check",
                        "Evidence Excerpt": ref.get("excerpt") or "",
                    }
                    for ref in ev_refs
                ]
                ui.dataframe(ref_rows, hide_index=True, width="stretch")
            elif ev_ids:
                ui.caption("**Supporting step(s):** " + ", ".join(f"`[{sid}]`" for sid in ev_ids))
            else:
                ui.caption("**Supporting evidence references:** None recorded for this criterion.")


def _render_structured_response(
    answer: str,
    completion: dict | None,
    trace_lines: list[str],
    *,
    msg_key: str = "live",
) -> None:
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

    _render_runtime_telemetry(completion)

    parsed = parse_structured_answer(answer)
    st.markdown(parsed["conclusion"] or "(No conclusion produced)")

    _render_mutation_diff_card(completion, msg_key=msg_key)
    _render_acceptance_criteria(completion)

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


def _render_chat_workspace(app, user_input: str | None) -> None:
    for idx_m, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else None):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_structured_response(
                    msg["content"],
                    msg.get("completion"),
                    msg.get("trace", []),
                    msg_key=f"hist_{idx_m}",
                )

    st.caption("Try a focused question; ARI retains multi-turn chat context, grades acceptance criteria, and verifies guarded edits.")
    cols = st.columns(len(EXAMPLE_QUERIES))
    triggered_query = None
    for i, example in enumerate(EXAMPLE_QUERIES):
        if cols[i].button(example, key=f"ari_example_{i}", width="stretch"):
            triggered_query = example
    active_query = triggered_query or user_input
    if not active_query:
        return

    prior_history = [
        {
            "role": m.get("role", "user"),
            "content": m.get("content", ""),
            "completion": m.get("completion"),
        }
        for m in st.session_state["messages"]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(active_query)
    with st.chat_message("assistant"):
        trace_lines: list[str] = []
        with st.status("Investigating repository…", expanded=True) as status_box:
            trace_placeholder = st.empty()
            event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
            result_holder: dict[str, object] = {}

            def on_trace(line: str) -> None:
                event_queue.put(("trace", line))

            def _worker() -> None:
                try:
                    answer, completion = run_query(
                        app.agent,
                        active_query,
                        on_line=on_trace,
                        service=app.service,
                        chat_history=prior_history,
                    )
                    result_holder["answer"] = answer
                    result_holder["completion"] = completion
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
                    trace_placeholder.markdown("  \n".join(_format_trace_badge(line) for line in trace_lines))
                elif kind == "done":
                    break
            worker_thread.join()
            if "error" in result_holder:
                status_box.update(label="Investigation failed", state="error", expanded=True)
                st.error(f"Error: {result_holder['error']}")
                return
            answer = str(result_holder.get("answer") or "")
            completion = result_holder.get("completion")
            support = completion.get("support", "SUPPORTED") if isinstance(completion, dict) else "SUPPORTED"
            status_box.update(label=f"Investigation complete · {support}", state="complete", expanded=False)
        st.session_state["messages"].append({"role": "user", "content": active_query})
        msg_idx = len(st.session_state["messages"])
        _render_structured_response(answer, completion, trace_lines, msg_key=f"hist_{msg_idx}")  # type: ignore[arg-type]
        st.session_state["messages"].append({
            "role": "assistant", "content": answer, "completion": completion, "trace": trace_lines,
        })


def main() -> None:
    st.set_page_config(
        page_title="ARI — Agentic Repository Intelligence",
        page_icon="🧭",
        layout="wide",
    )
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root { --ari-ink: #e8edf7; --ari-muted: #93a0b8; --ari-line: rgba(146,164,197,.16); --ari-accent: #79a9ff; }
      .stApp { background: radial-gradient(ellipse at 78% -15%, rgba(77,117,191,.16), transparent 34%), #0b1019; color: var(--ari-ink); }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
      h1 { font-size: 2.35rem !important; }
      [data-testid="stSidebar"] { background: #0d1420; border-right: 1px solid var(--ari-line); }
      [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
      [data-testid="stMetric"] { background: rgba(21,31,47,.72); border: 1px solid var(--ari-line); border-radius: 14px; padding: 14px 16px; }
      [data-testid="stMetricLabel"] { color: var(--ari-muted); }
      [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
      [data-testid="stTabs"] button { font-weight: 600; }
      [data-testid="stChatMessage"] { border: 1px solid var(--ari-line); border-radius: 16px; padding: 1rem; background: rgba(16,24,37,.62); }
      div[data-testid="stExpander"] { border: 1px solid var(--ari-line); border-radius: 12px; background: rgba(16,24,37,.45); }
      .ari-eyebrow { color: #8fb6ff; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
      .ari-subtitle { color: var(--ari-muted); margin-top: -.65rem; }
      div[data-testid="stButton"] button[kind="primary"] { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "ari_app" not in st.session_state:
        st.session_state["ari_app"] = None
    if "local_repo_input" not in st.session_state:
        st.session_state["local_repo_input"] = ""
    if "local_idx_input" not in st.session_state:
        st.session_state["local_idx_input"] = ""

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
                    placeholder=r"D:\projects\my-repo",
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
                    placeholder="e.g. data/indices/my-repo (auto-created if missing)",
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

        else:
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

    app = st.session_state.get("ari_app")
    if app is None:
        st.markdown('<div class="ari-eyebrow">Agentic Repository Intelligence</div>', unsafe_allow_html=True)
        st.title("Understand your codebase.")
        st.markdown('<p class="ari-subtitle">Load a repository to ask evidence-backed questions, inspect its structure, and review its indexed source.</p>', unsafe_allow_html=True)
        st.info("Choose a local folder, public GitHub repository, or cached repository from the sidebar to open your workspace.")
        st.markdown("**ARI workflow**  \n`Repository` → `AST + text index` → `Retrieval + relationship graph` → `Bounded agent` → `Evidence-backed answer`")
        return

    idx = app.service.index
    repo_name = Path(idx.root).name or idx.root
    st.markdown('<div class="ari-eyebrow">Repository workspace</div>', unsafe_allow_html=True)
    st.title(repo_name)
    st.markdown(f'<p class="ari-subtitle">{idx.root}</p>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Python files", len(idx.files))
    metric_cols[1].metric("Symbols", len(idx.symbols))
    metric_cols[2].metric("Relationships", len(idx.edges))
    metric_cols[3].metric("Search chunks", len(idx.chunks))
    retriever_name = getattr(app.service.retriever, "backend_name", type(app.service.retriever).__name__)
    st.caption(f"Index: `{st.session_state.get('ari_index_dir') or 'in memory'}`  ·  Retrieval: `{retriever_name}`  ·  Model: `{app.config.model}`")

    workspace = st.radio(
        "Workspace",
        options=["Ask ARI", "Explore repository"],
        horizontal=True,
        key="ari_workspace",
        label_visibility="collapsed",
    )
    user_input = st.chat_input("Ask ARI about this repository…")
    if user_input and workspace != "Ask ARI":
        st.session_state["ari_workspace"] = "Ask ARI"
        workspace = "Ask ARI"

    if workspace == "Explore repository":
        render_repository_explorer(st, app)
    else:
        _render_chat_workspace(app, user_input)


if __name__ == "__main__":
    main()
