"""Core data model: skills as first-class, versioned artifacts.

概念模型（见《SkillOps开源项目提案》§2）:

    Trajectory ──distill──> Skill ──gate──> SkillVersion ──register──> Registry
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SkillStatus(str, enum.Enum):
    """Lifecycle states of a skill version."""

    STAGED = "staged"        # 候选区：已登记，未过门控
    COMMITTED = "committed"  # 已入库：不可变快照
    REJECTED = "rejected"    # 门控驳回
    RETIRED = "retired"      # 生命周期策略退役
    MERGED = "merged"        # 被合并进其他版本


@dataclass
class SkillVersion:
    """An immutable snapshot of a skill at one point in its lineage."""

    skill_id: str                       # stable id across versions (name slug)
    version: int                        # monotonically increasing per skill_id
    name: str                           # human-readable name
    rel_path: str                       # path inside the repo's skills/ dir
    sha256: str                         # content hash
    status: SkillStatus = SkillStatus.STAGED
    parent_ids: list[str] = field(default_factory=list)           # lineage pointers (vids)
    source_trajectories: list[str] = field(default_factory=list)  # provenance
    generator: str = ""                 # which evolver produced it (e.g. "evoskill-v0.3")
    metrics: dict = field(default_factory=dict)                   # gate / task metrics
    created_at: str = field(default_factory=utc_now)

    @property
    def vid(self) -> str:
        """Global version id, e.g. 'refund_triage@v3'."""
        return f"{self.skill_id}@v{self.version}"
