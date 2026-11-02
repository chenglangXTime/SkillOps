"""Adapter for the Claude Code / OpenClaw SKILL.md ecosystem.

M1 scope: discover SKILL.md files under a directory so they can be staged
into a SkillOps registry in bulk. Runtime integration hooks arrive in M3.
"""

from __future__ import annotations

from pathlib import Path


def discover_skills(root: str | Path) -> list[Path]:
    """Find all SKILL.md files under `root` (recursive, case-insensitive name)."""
    root = Path(root)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.name.lower() == "skill.md"
    )
