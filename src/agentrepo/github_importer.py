#!/usr/bin/env python3
"""Public GitHub repository importer and indexer for ARI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable
import urllib.error
import urllib.request
import zipfile


def find_project_root() -> Path:
    """Locate the project root directory regardless of which subfolder this script lives in."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (
            (candidate / "src" / "agentrepo").is_dir()
            or (candidate / "main.py").is_file()
            or (candidate / ".env").is_file()
        ):
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
DEFAULT_REPOS_DIR = PROJECT_ROOT / "data" / "repos"
DEFAULT_INDICES_DIR = PROJECT_ROOT / "data" / "indices"

for _cand in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from agentrepo import index_repository, save_index  # noqa: E402


_GITHUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?"
    r"(?:/(?:tree|blob|commits)/([^/?#]+))?"
    r"(?:/.*)?$",
    re.IGNORECASE,
)
_SLUG_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitHubImportError(Exception):
    """Raised when a GitHub URL is invalid or cannot be cloned/indexed."""


@dataclass(frozen=True)
class GitHubRepoSpec:
    owner: str
    repo: str
    branch: str | None = None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def folder_name(self) -> str:
        suffix = f"__{self.branch}" if self.branch else ""
        safe_suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", suffix)
        return f"{self.owner}__{self.repo}{safe_suffix}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


@dataclass(frozen=True)
class ImportedRepo:
    spec: GitHubRepoSpec
    repo_dir: Path
    index_dir: Path | None
    file_count: int
    python_file_count: int
    symbol_count: int
    edge_count: int
    chunk_count: int
    reused_cached_clone: bool


def _rmtree_windows_safe(path: Path) -> None:
    """Remove a directory tree safely on Windows even when `.git` packfiles are read-only."""
    if not path.exists():
        return

    def _handle_remove_readonly(func, target_path, exc_info):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            func(target_path)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_handle_remove_readonly)


def parse_github_url(url_or_slug: str, branch: str | None = None) -> GitHubRepoSpec:
    """Parse and validate a public GitHub URL or `owner/repo` shorthand."""
    raw = (url_or_slug or "").strip().rstrip("/")
    if not raw:
        raise GitHubImportError("GitHub repository URL or 'owner/repo' cannot be empty.")

    m = _GITHUB_URL_RE.match(raw)
    url_branch: str | None = None
    if m:
        owner = m.group("owner")
        repo = m.group("repo")
        url_branch = m.group(3)
    else:
        m_slug = _SLUG_RE.match(raw)
        if not m_slug:
            raise GitHubImportError(
                f"Invalid GitHub URL or slug: {raw!r}. "
                "Expected 'https://github.com/<owner>/<repo>' or '<owner>/<repo>'."
            )
        owner = m_slug.group("owner")
        repo = m_slug.group("repo")

    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    eff_branch = (branch or url_branch or "").strip() or None

    if (
        not _SAFE_NAME_RE.match(owner)
        or not _SAFE_NAME_RE.match(repo)
        or owner in {".", ".."}
        or repo in {".", ".."}
    ):
        raise GitHubImportError(f"Unsafe owner or repository name in: {raw!r}")

    return GitHubRepoSpec(owner=owner, repo=repo, branch=eff_branch)


def _clone_with_git(spec: GitHubRepoSpec, target_dir: Path, timeout_sec: int = 180) -> None:
    """Run a shallow `git clone --depth 1` into `target_dir`."""
    git_bin = shutil.which("git")
    if not git_bin:
        raise FileNotFoundError("git binary not found on PATH")

    cmd = [git_bin, "clone", "--depth", "1", "--single-branch"]
    if spec.branch:
        cmd.extend(["--branch", spec.branch])
    cmd.extend([spec.clone_url, str(target_dir)])

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise GitHubImportError(f"git clone failed for {spec.slug}: {stderr}")


def _download_zip_fallback(spec: GitHubRepoSpec, target_dir: Path, timeout_sec: int = 180) -> None:
    """Fallback downloader using GitHub ZIP archive when `git` is unavailable."""
    candidates = [spec.branch] if spec.branch else ["main", "master"]
    last_err: Exception | None = None

    for ref in candidates:
        zip_url = f"https://codeload.github.com/{spec.owner}/{spec.repo}/zip/refs/heads/{ref}"
        req = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "ARI-GitHub-Importer/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                payload = resp.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                with tempfile.TemporaryDirectory() as tmp:
                    zf.extractall(tmp)
                    extracted_roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
                    src_root = extracted_roots[0] if len(extracted_roots) == 1 else Path(tmp)
                    if target_dir.exists():
                        _rmtree_windows_safe(target_dir)
                    shutil.copytree(src_root, target_dir)
            return
        except Exception as err:
            last_err = err

    raise GitHubImportError(
        f"Could not download GitHub archive for {spec.slug}: {last_err}"
    ) from last_err


def resolve_storage_dir(path_val: str | Path | None, default_dir: Path) -> Path:
    """Resolve a storage path relative to `PROJECT_ROOT` if not already absolute."""
    if path_val is None or not str(path_val).strip():
        return default_dir.resolve()
    p = Path(str(path_val).strip()).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def ensure_local_index(
    repo_dir: str | Path,
    index_dir: str | Path | None = None,
    *,
    force_rebuild: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Ensure a saved index exists for `repo_dir` (creating it automatically if missing).

    If `index_dir` is None or empty, defaults to `<project_root>/data/indices/<repo_folder_name>`.
    """
    def notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    root = Path(repo_dir).expanduser().resolve()
    if not root.is_dir():
        raise GitHubImportError(f"Repository path does not exist or is not a directory: {root}")

    if index_dir is None or not str(index_dir).strip():
        target_index = DEFAULT_INDICES_DIR / root.name
    else:
        raw_idx = Path(str(index_dir).strip()).expanduser()
        target_index = raw_idx.resolve() if raw_idx.is_absolute() else (PROJECT_ROOT / raw_idx).resolve()

    has_saved_files = (
        target_index.is_dir()
        and (target_index / "symbols.json").is_file()
        and (target_index / "graph.json").is_file()
        and (target_index / "chunks.json").is_file()
    )

    if not has_saved_files or force_rebuild:
        notify(f"Building index for {root.name} -> {target_index} ...")
        target_index.mkdir(parents=True, exist_ok=True)
        index, _ = index_repository(root)
        save_index(index, target_index)
        notify(
            f"Saved index ({len(index.symbols)} symbols, {len(index.chunks)} chunks) to {target_index}"
        )
    return target_index


def import_github_repo(
    url_or_slug: str,
    *,
    repos_dir: str | Path | None = None,
    indices_dir: str | Path | None = None,
    branch: str | None = None,
    force_refresh: bool = False,
    build_saved_index: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> ImportedRepo:
    """Clone (or reuse) a public GitHub repository into `<project_root>/data/repos/`
    and build its ARI index into `<project_root>/data/indices/`."""
    def notify(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    spec = parse_github_url(url_or_slug, branch=branch)
    repos_root = resolve_storage_dir(repos_dir, DEFAULT_REPOS_DIR)
    indices_root = resolve_storage_dir(indices_dir, DEFAULT_INDICES_DIR)
    repos_root.mkdir(parents=True, exist_ok=True)
    if build_saved_index:
        indices_root.mkdir(parents=True, exist_ok=True)

    target_repo_dir = repos_root / spec.folder_name
    target_index_dir = (indices_root / spec.folder_name) if build_saved_index else None

    reused_clone = False
    if target_repo_dir.is_dir() and any(target_repo_dir.iterdir()) and not force_refresh:
        reused_clone = True
        notify(f"Reusing cached repository at {target_repo_dir}")
    else:
        if target_repo_dir.exists():
            notify(f"Removing existing clone at {target_repo_dir} for fresh import...")
            _rmtree_windows_safe(target_repo_dir)
        notify(f"Cloning {spec.html_url}" + (f" (branch: {spec.branch})" if spec.branch else "") + " ...")
        try:
            _clone_with_git(spec, target_repo_dir)
        except FileNotFoundError:
            notify("`git` not found on PATH; falling back to GitHub ZIP archive download...")
            _download_zip_fallback(spec, target_repo_dir)

    notify(f"Indexing repository files in {target_repo_dir.name}...")
    try:
        index, _ = index_repository(target_repo_dir)
        if target_index_dir is not None:
            target_index_dir.mkdir(parents=True, exist_ok=True)
            save_index(index, target_index_dir)
            notify(f"Saved index to {target_index_dir}")
    except Exception as exc:
        raise GitHubImportError(f"Failed to index {spec.slug}: {exc}") from exc

    return ImportedRepo(
        spec=spec,
        repo_dir=target_repo_dir,
        index_dir=target_index_dir,
        file_count=len(index.repo_files or []),
        python_file_count=len(index.files),
        symbol_count=len(index.symbols),
        edge_count=len(index.edges),
        chunk_count=len(index.chunks),
        reused_cached_clone=reused_clone,
    )


def list_cached_repos(
    repos_dir: str | Path | None = None,
    indices_dir: str | Path | None = None,
) -> list[tuple[str, Path, Path | None]]:
    """Return `(name, repo_path, index_path_or_None)` for locally cached repos in `data/repos/`."""
    repos_root = resolve_storage_dir(repos_dir, DEFAULT_REPOS_DIR)
    indices_root = resolve_storage_dir(indices_dir, DEFAULT_INDICES_DIR)
    if not repos_root.is_dir():
        return []

    items: list[tuple[str, Path, Path | None]] = []
    for child in sorted(repos_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            idx_candidate = indices_root / child.name
            has_idx = idx_candidate.is_dir() and (idx_candidate / "symbols.json").is_file()
            items.append((child.name, child, idx_candidate if has_idx else None))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import and index a public GitHub repository for ARI.")
    parser.add_argument("url", help="Public GitHub repository URL or 'owner/repo' slug")
    parser.add_argument("--branch", help="Optional branch or tag to clone")
    parser.add_argument("--repos-dir", default=None, help="Directory to store cloned repositories (default: <root>/data/repos)")
    parser.add_argument("--indices-dir", default=None, help="Directory to store built ARI indices (default: <root>/data/indices)")
    parser.add_argument("--force", action="store_true", help="Re-clone even if cached locally")
    args = parser.parse_args(argv)

    try:
        res = import_github_repo(
            args.url,
            repos_dir=args.repos_dir,
            indices_dir=args.indices_dir,
            branch=args.branch,
            force_refresh=args.force,
            on_progress=lambda msg: print(f"[GitHubImporter] {msg}", flush=True),
        )
    except GitHubImportError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2

    print("\nImport Complete!")
    print(f"  Repository : {res.spec.slug} -> {res.repo_dir}")
    print(f"  Index Dir  : {res.index_dir}")
    print(
        f"  Stats      : {res.file_count} files ({res.python_file_count} Python), "
        f"{res.symbol_count} symbols, {res.edge_count} edges, {res.chunk_count} chunks"
    )
    print(f"\nRun CLI with:\n  python main.py --repo \"{res.repo_dir}\" --index-dir \"{res.index_dir}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
