"""Gate: evaluation checkpoint before a staged skill enters the library.

M2 交付：分层门控（静态检查 → 小样本快筛 → 完整基准）+ 锁定 holdout。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateVerdict:
    passed: bool
    metrics: dict = field(default_factory=dict)
    reason: str = ""


class Gate:
    """Not implemented in M1 — see proposal §4.1 for the target API."""

    def __init__(self, benchmark: str, baseline: str = "current_library",
                 min_delta: float = 0.0, holdout: str = "locked_split"):
        self.benchmark = benchmark
        self.baseline = baseline
        self.min_delta = min_delta
        self.holdout = holdout

    def evaluate(self, candidate) -> GateVerdict:  # pragma: no cover - M2
        raise NotImplementedError(
            "Gate evaluation ships in M2. In M1, use `skillops commit` as a manual gate."
        )
