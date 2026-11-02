"""Experiment runner: evolution loop + run manifest.

一次实验 = 一个不可变的 RunManifest：

    runs/<run_id>/manifest.json
    ├── 配置快照（benchmark / evolver / arm / rounds / seed / min_delta）
    ├── 代码版本与 git HEAD
    ├── 每轮记录（候选 vid、决策、评估集准确率、库大小）
    ├── 最终指标（评估集 + 锁定 holdout，holdout 全程只评一次）
    └── 成本（LLM token 数；scripted 为 0）

实验臂（对应论文 RQ2 消融）：
- `ungoverned`：所有候选直接入库 —— RQ1 退化对照组
- `gate`：候选必须过三层门控 —— 治理臂

锁定纪律：holdout split 只在实验结束时评一次，过程中只看 eval split。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import skillops
from skillops.core.models import utc_now
from skillops.core.registry import Registry
from skillops.gate.benchmark import Benchmark
from skillops.gate.engine import Gate
from skillops.experiments.evolvers import make_evolver


@dataclass
class ExperimentConfig:
    benchmark: str
    evolver: str = "scripted"
    arm: str = "ungoverned"          # "ungoverned" | "gate"
    rounds: int = 10
    seed: int = 42
    min_delta: float = 0.0
    out: str | None = None           # default: <repo>/runs


def _eval_library(reg: Registry, bench: Benchmark, task_ids: list[str]) -> float:
    view = reg.library_view()
    try:
        return bench.run(view, task_ids)["accuracy"]
    finally:
        shutil.rmtree(view, ignore_errors=True)


def run_experiment(reg: Registry, config: ExperimentConfig) -> dict:
    bench = Benchmark.load(config.benchmark)
    evolver = make_evolver(config.evolver, seed=config.seed)
    gate = Gate(bench, min_delta=config.min_delta) if config.arm == "gate" else None
    eval_ids = bench.eval_split()

    run_id = (
        f"{utc_now().replace(':', '').replace('+', 'Z')}"
        f"_{config.arm}_{config.evolver}_s{config.seed}"
    )
    out_dir = Path(config.out) if config.out else reg.root / "runs"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "run_id": run_id,
        "created_at": utc_now(),
        "config": asdict(config),
        "code_version": skillops.__version__,
        "git_head_start": reg.git.head(),
        "benchmark": {"name": bench.name, "eval_tasks": len(eval_ids),
                      "holdout_tasks": len(bench.holdout_split())},
        "rounds": [],
        "final": {},
        "cost": {"llm_tokens": 0},
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="skillops_exp_"))
    try:
        for r in range(1, config.rounds + 1):
            name, content = evolver.propose(r)
            skill_file = tmp_dir / f"{name}.md"
            skill_file.write_text(content, encoding="utf-8")
            v = reg.stage(skill_file, generator=f"{config.evolver}-evolver")

            if gate is not None:
                verdict = gate.evaluate(reg, v.vid)
                if verdict.passed:
                    reg.commit(v.vid)
                    decision = "committed"
                else:
                    reg.reject(v.vid, reason=verdict.reason)
                    decision = "rejected"
            else:
                reg.commit(v.vid)
                decision = "committed"

            acc = _eval_library(reg, bench, eval_ids)
            manifest["rounds"].append({
                "round": r,
                "vid": v.vid,
                "decision": decision,
                "eval_accuracy": acc,
                "library_size": len(
                    [x for x in reg.list_versions()
                     if x.status.value == "committed"]
                ),
            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Locked holdout: evaluated exactly once, at the end.
    manifest["final"] = {
        "eval_accuracy": manifest["rounds"][-1]["eval_accuracy"]
        if manifest["rounds"] else 0.0,
        "peak_eval_accuracy": max(
            (x["eval_accuracy"] for x in manifest["rounds"]), default=0.0
        ),
        "holdout_accuracy": _eval_library(reg, bench, bench.holdout_split()),
        "library_size": manifest["rounds"][-1]["library_size"]
        if manifest["rounds"] else 0,
    }
    manifest["git_head_end"] = reg.git.head()

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reg.ledger.audit("experiment", {
        "run_id": run_id, "arm": config.arm, "evolver": config.evolver,
        "seed": config.seed, "rounds": config.rounds,
        "final_eval": manifest["final"]["eval_accuracy"],
    })
    return manifest
