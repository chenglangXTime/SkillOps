"""Registry: the governance ledger over the skill repo.

Registry 把 git（技能内容存储）与 SQLite 账本（谱系、状态、指标、审计）
组合成一个「可 diff、可审计、可回滚」的技能库入口。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from skillops.core.models import SkillStatus, SkillVersion, utc_now
from skillops.store.git_backend import GitBackend
from skillops.store.ledger import Ledger

SKILLOPS_DIR = ".skillops"
SKILLS_DIR = "skills"


class RegistryError(RuntimeError):
    pass


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "unnamed"


class Registry:
    """Entry point: `reg = Registry(root)` opens, `Registry.init(root)` creates."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        meta = self.root / SKILLOPS_DIR
        if not meta.exists():
            raise RegistryError(
                f"{self.root} is not a skillops repo. Run `skillops init` first."
            )
        self.ledger = Ledger(meta / "ledger.db")
        self.git = GitBackend(self.root)

    # ---- lifecycle ----

    @classmethod
    def init(cls, root: str | Path) -> "Registry":
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        meta = root / SKILLOPS_DIR
        if meta.exists():
            raise RegistryError(f"{root} is already a skillops repo.")
        (root / SKILLS_DIR).mkdir(parents=True)
        (root / "trajectories").mkdir()
        meta.mkdir()
        (meta / "config.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "created_at": utc_now(),
                    "gate": {"benchmark": None, "min_delta": 0.0},
                    "policies": {"retirement": "default", "merge": "default"},
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        git = GitBackend(root)
        git.init()
        reg = cls(root)
        reg.ledger.audit("init", {"root": str(root)})
        git.commit_all("skillops: init skill repo")
        return reg

    # ---- staging & commit ----

    def stage(
        self,
        skill_path: str | Path,
        source_trajectories: list[str] | None = None,
        generator: str = "",
    ) -> SkillVersion:
        """Register a candidate skill into the staging area (not yet gated)."""
        src = Path(skill_path)
        if not src.is_file():
            raise RegistryError(f"skill file not found: {src}")

        content = src.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        name = src.stem
        skill_id = _slugify(name)
        version = self.ledger.next_version(skill_id)

        rel = Path(SKILLS_DIR) / f"{skill_id}.v{version}{src.suffix}"
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

        v = SkillVersion(
            skill_id=skill_id,
            version=version,
            name=name,
            rel_path=str(rel),
            sha256=digest,
            status=SkillStatus.STAGED,
            parent_ids=self.ledger.latest_parents(skill_id),
            source_trajectories=list(source_trajectories or []),
            generator=generator,
        )
        self.ledger.insert_version(v)
        self.ledger.audit(
            "stage",
            {"vid": v.vid, "sha256": digest, "generator": generator},
        )
        return v

    def commit(self, vid: str, metrics: dict | None = None) -> str:
        """Promote a staged version into the library (M1: manual gate)."""
        v = self.ledger.get(vid)
        if v is None:
            raise RegistryError(f"unknown version: {vid}")
        if v.status is not SkillStatus.STAGED:
            raise RegistryError(f"{vid} is {v.status.value}, not staged")

        # Retire the previously committed version of the same skill, if any.
        for old_vid in v.parent_ids:
            old = self.ledger.get(old_vid)
            if old and old.status is SkillStatus.COMMITTED:
                self.ledger.update_status(old_vid, SkillStatus.RETIRED)
                self.ledger.audit("retire", {"vid": old_vid, "reason": "superseded"})

        self.ledger.update_status(vid, SkillStatus.COMMITTED, metrics or v.metrics)
        commit_hash = self.git.commit_all(f"skillops: commit {vid}")
        self.ledger.audit("commit", {"vid": vid, "git": commit_hash})
        return commit_hash or "(no changes)"

    def reject(self, vid: str, reason: str = "") -> None:
        v = self.ledger.get(vid)
        if v is None:
            raise RegistryError(f"unknown version: {vid}")
        self.ledger.update_status(vid, SkillStatus.REJECTED)
        self.ledger.audit("reject", {"vid": vid, "reason": reason})

    # ---- inspection ----

    def status(self) -> dict:
        versions = self.ledger.list_versions()
        by_status: dict[str, int] = {}
        for v in versions:
            by_status[v.status.value] = by_status.get(v.status.value, 0) + 1
        return {
            "root": str(self.root),
            "git_head": self.git.head(),
            "total_versions": len(versions),
            "by_status": by_status,
        }

    def log(self, limit: int = 20) -> list[dict]:
        return self.ledger.events(limit=limit)

    def list_versions(self, status: SkillStatus | None = None) -> list[SkillVersion]:
        return self.ledger.list_versions(status)
