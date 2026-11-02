"""Lifecycle policies: retirement, merge, drift detection. (M2)

默认实现对应 Ratchet 式生命周期纪律：退役、容量约束、元技能引导。
"""

from __future__ import annotations

from typing import Protocol

from skillops.core.models import SkillVersion


class RetirementPolicy(Protocol):
    def should_retire(self, skill: SkillVersion, history: list[dict]) -> bool: ...


class MergePolicy(Protocol):
    def cluster(self, skills: list[SkillVersion]) -> list[list[SkillVersion]]: ...


class Policy:
    """Not implemented in M1 — see proposal §4.3 for the target API."""

    @classmethod
    def default(cls) -> "Policy":  # pragma: no cover - M2
        raise NotImplementedError("Lifecycle policies ship in M2.")

    def sweep(self, registry):  # pragma: no cover - M2
        raise NotImplementedError("Lifecycle policies ship in M2.")
