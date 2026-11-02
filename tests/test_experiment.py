"""Experiment harness tests: RQ1 feasibility (drift) + RQ2 arm (gate resists)."""

import json
from pathlib import Path

import pytest

from skillops.core.registry import Registry
from skillops.experiments.runner import ExperimentConfig, run_experiment

BENCH = str(Path(__file__).resolve().parent.parent / "benchmarks" / "officeqa_mini")


@pytest.fixture()
def make_repo(tmp_path: Path):
    def _make(name: str) -> Registry:
        return Registry.init(tmp_path / name)
    return _make


def test_ungoverned_arm_shows_drift(make_repo):
    """RQ1 可行性：无治理臂下，检索干扰导致技能库准确率随轮次退化。"""
    reg = make_repo("ungoverned")
    m = run_experiment(reg, ExperimentConfig(
        benchmark=BENCH, arm="ungoverned", rounds=10, seed=7,
    ))

    accs = [r["eval_accuracy"] for r in m["rounds"]]
    peak = m["final"]["peak_eval_accuracy"]
    final = m["final"]["eval_accuracy"]
    assert peak == pytest.approx(1.0)          # 有用技能入库后达到满点
    assert final < peak - 0.3                  # 退化显著发生
    assert accs[-1] == pytest.approx(final)
    # 库在无治理下持续膨胀
    assert m["final"]["library_size"] == 10

    # run manifest 落盘且含关键字段
    manifest_path = reg.root / "runs" / m["run_id"] / "manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["config"]["seed"] == 7
    assert saved["config"]["arm"] == "ungoverned"
    assert saved["cost"]["llm_tokens"] == 0
    assert len(saved["rounds"]) == 10
    # holdout 纪律：只在 final 出现一次
    assert "holdout_accuracy" in saved["final"]


def test_gate_arm_resists_drift(make_repo):
    """RQ2 治理臂：同一 evolver，门控拦截退化技能，准确率保持满点。"""
    reg = make_repo("gated")
    m = run_experiment(reg, ExperimentConfig(
        benchmark=BENCH, arm="gate", rounds=10, seed=7,
    ))

    assert m["final"]["eval_accuracy"] == pytest.approx(1.0)
    assert m["final"]["holdout_accuracy"] == pytest.approx(1.0)
    decisions = [r["decision"] for r in m["rounds"]]
    assert "rejected" in decisions              # 退化技能被拦截
    assert m["final"]["library_size"] < 10      # 库未膨胀


def test_same_seed_same_trajectory(make_repo):
    """确定性：相同 seed 的两臂实验，候选轨迹完全一致。"""
    m1 = run_experiment(make_repo("a"), ExperimentConfig(
        benchmark=BENCH, arm="ungoverned", rounds=6, seed=99))
    m2 = run_experiment(make_repo("b"), ExperimentConfig(
        benchmark=BENCH, arm="ungoverned", rounds=6, seed=99))
    assert [r["vid"] for r in m1["rounds"]] == [r["vid"] for r in m2["rounds"]]
    assert [r["eval_accuracy"] for r in m1["rounds"]] == \
           [r["eval_accuracy"] for r in m2["rounds"]]
