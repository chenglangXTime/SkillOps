"""Gate: layered evaluation checkpoint before a staged skill enters the library.

三层门控（提案 §4.1 / §7 的"评测即一等公民"）：

1. **静态检查**：格式、长度、密钥泄露、占位符 —— 零成本，fail-fast
2. **小样本快筛**：在 quick_sample 上对比 baseline / candidate，明显回退提前驳回
3. **完整基准 + 锁定 holdout 认证**：全量评估 split 上算 delta，
   最后在 locked holdout 上做一次认证，双重确认增益可迁移、非过拟合

所有指标写入账本（`skillops log` 可审计），未通过门控的技能只能留在 staging。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field

from skillops.core.models import SkillStatus
from skillops.core.registry import Registry, RegistryError
from skillops.gate.benchmark import Benchmark
from skillops.gate.static_checks import has_blocker, run_static_checks


@dataclass
class GateVerdict:
    passed: bool
    metrics: dict = field(default_factory=dict)
    reason: str = ""


class Gate:
    def __init__(
        self,
        benchmark: str | Benchmark,
        min_delta: float = 0.0,
        quick_tolerance: float = 0.0,
        eval_fn=None,
        certify_holdout: bool = False,
    ):
        self.benchmark = (
            benchmark if isinstance(benchmark, Benchmark) else Benchmark.load(benchmark)
        )
        self.min_delta = min_delta
        self.quick_tolerance = quick_tolerance
        # eval_fn(library_dir, task_ids) -> {"scores":..., "accuracy":...}
        # 默认 None → 使用基准自带的 subprocess harness；LLM 实验注入 LLM 评估器。
        self.eval_fn = eval_fn
        # 锁定集纪律：常规入库决策只用 static + quick + eval 三层；
        # holdout 认证是显式的一次性动作（如版本发布前），避免对锁定集的
        # 自适应使用（adaptive overfitting / 信息泄漏）。
        self.certify_holdout = certify_holdout

    def _run(self, library_dir, task_ids) -> dict:
        if self.eval_fn is not None:
            return self.eval_fn(library_dir, task_ids)
        return self.benchmark.run(library_dir, task_ids)

    def evaluate(self, registry: Registry, vid: str) -> GateVerdict:
        v = registry.ledger.get(vid)
        if v is None:
            raise RegistryError(f"unknown version: {vid}")
        if v.status is not SkillStatus.STAGED:
            raise RegistryError(f"{vid} is {v.status.value}, not staged")

        metrics: dict = {"benchmark": self.benchmark.name, "layers": {}}

        # ---- Layer 1: static checks ----
        content = (registry.root / v.rel_path).read_text(encoding="utf-8")
        findings = run_static_checks(content)
        metrics["layers"]["static"] = {
            "findings": [f.__dict__ for f in findings],
            "passed": not has_blocker(findings),
        }
        if has_blocker(findings):
            verdict = GateVerdict(
                False, metrics,
                "static checks failed: "
                + "; ".join(f.message for f in findings if f.severity == "block"),
            )
            registry.record_gate_result(vid, verdict.passed, metrics)
            return verdict

        # Prepare library views: baseline (committed only) vs candidate swapped in.
        baseline_dir = registry.library_view()
        candidate_dir = registry.library_view(candidate_vid=vid)
        bench = self.benchmark
        try:
            # ---- Layer 2: quick screen ----
            quick_ids = bench.quick_split()
            base_quick = self._run(baseline_dir, quick_ids)
            cand_quick = self._run(candidate_dir, quick_ids)
            quick_delta = cand_quick["accuracy"] - base_quick["accuracy"]
            metrics["layers"]["quick"] = {
                "tasks": quick_ids,
                "baseline": base_quick["accuracy"],
                "candidate": cand_quick["accuracy"],
                "delta": quick_delta,
            }
            if quick_delta < -self.quick_tolerance:
                verdict = GateVerdict(
                    False, metrics,
                    f"quick screen regression: delta {quick_delta:+.3f} "
                    f"< -{self.quick_tolerance:.3f}",
                )
                registry.record_gate_result(vid, verdict.passed, metrics)
                return verdict

            # ---- Layer 3a: full evaluation split ----
            eval_ids = bench.eval_split()
            base_full = self._run(baseline_dir, eval_ids)
            cand_full = self._run(candidate_dir, eval_ids)
            full_delta = cand_full["accuracy"] - base_full["accuracy"]
            metrics["layers"]["full"] = {
                "tasks": len(eval_ids),
                "baseline": base_full["accuracy"],
                "candidate": cand_full["accuracy"],
                "delta": full_delta,
            }

            # ---- Layer 3b (optional): locked holdout certification ----
            hold_ids = bench.holdout_split()
            hold_delta = 0.0
            if hold_ids and self.certify_holdout:
                base_hold = self._run(baseline_dir, hold_ids)
                cand_hold = self._run(candidate_dir, hold_ids)
                hold_delta = cand_hold["accuracy"] - base_hold["accuracy"]
                metrics["layers"]["holdout"] = {
                    "tasks": len(hold_ids),
                    "baseline": base_hold["accuracy"],
                    "candidate": cand_hold["accuracy"],
                    "delta": hold_delta,
                }
        finally:
            shutil.rmtree(baseline_dir, ignore_errors=True)
            shutil.rmtree(candidate_dir, ignore_errors=True)

        passed = full_delta >= self.min_delta
        if "holdout" in metrics["layers"]:
            passed = passed and hold_delta >= self.min_delta
        reason = (
            f"full delta {full_delta:+.3f}"
            + (f", holdout delta {hold_delta:+.3f}" if "holdout" in metrics["layers"] else "")
            + f" (required >= {self.min_delta:+.3f})"
        )
        verdict = GateVerdict(passed, metrics, reason)
        registry.record_gate_result(vid, verdict.passed, metrics)
        return verdict
