"""Thin git wrapper — the skill repo's storage engine (no third-party deps)."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _run(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise GitError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


class GitBackend:
    def __init__(self, repo: Path):
        self.repo = Path(repo)

    def init(self) -> None:
        if not (self.repo / ".git").exists():
            _run(self.repo, "init")
        # Ensure commits work out of the box (repo-local identity only).
        if not _run(self.repo, "config", "user.name", check=False):
            _run(self.repo, "config", "user.name", "skillops")
        if not _run(self.repo, "config", "user.email", check=False):
            _run(self.repo, "config", "user.email", "skillops@localhost")

    def commit_all(self, message: str) -> str | None:
        """Commit every pending change; return short hash, or None if nothing to commit."""
        _run(self.repo, "add", "-A")
        if not _run(self.repo, "status", "--porcelain"):
            return None
        _run(self.repo, "commit", "-m", message)
        return _run(self.repo, "rev-parse", "--short", "HEAD")

    def head(self) -> str:
        return _run(self.repo, "rev-parse", "--short", "HEAD", check=False) or "(no commits)"
