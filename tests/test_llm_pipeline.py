"""LLM pipeline tests with a FakeClient (no network, deterministic)."""

import json
from pathlib import Path

import pytest

from skillops.core.registry import Registry
from skillops.experiments.llm_pipeline import (
    LLMExperimentConfig,
    answer_question,
    judge,
    retrieve_topk,
    run_llm_experiment,
)

BENCH = str(Path(__file__).resolve().parent.parent / "benchmarks" / "officeqa_mini")


class FakeClient:
    """Deterministic stand-in: answers correctly iff the gold answer's fact
    appears in the provided context (i.e., the skill library carries it);
    'revise' simply accumulates gold answers into a skill doc."""

    def __init__(self):
        self.model = "fake-model"
        self.usage = {"calls": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, max_tokens=1024, temperature=None):
        self.usage["calls"] += 1
        user = messages[-1]["content"]
        if "执行轨迹" in user:  # evolver revise call
            facts = "\n".join(
                f"- 事实：{line.split('正确答案：')[1]}"
                for line in user.splitlines() if "正确答案：" in line
            )
            return "# 蒸馏技能\n\n" + facts + "\n\n补充说明：以上为核实事实。"
        # answer call: echo any gold-like fact found in context
        for gold in ("9:30-18:30", "10号", "300元", "120人", "50元", "5天",
                     "5000元", "3天", "14天", "800元", "8848", "50000元"):
            if f"：{gold}" in user or f" {gold} " in user or gold in user.split("问题：")[0]:
                return f"答案是 {gold}"
        return "我不知道"


def test_judge():
    assert judge("5000元", "报销超过 5000元 需要审批") == 1
    assert judge("5000元", "报销超过 5000 元 需要审批") == 1   # 空白不敏感
    assert judge("300元", "上限为 300 元/人") == 1
    assert judge("5000元", "3000元") == 0


def test_retrieve_topk(tmp_path: Path):
    (tmp_path / "a.md").write_text("# 报销\n差旅报销单超过 5000元 需要总监审批。")
    (tmp_path / "b.md").write_text("# 请假\n病假超过 3天 需要三甲医院证明。")
    hits = retrieve_topk("差旅报销单超过多少元需要总监审批？", tmp_path, k=1)
    assert len(hits) == 1 and "5000元" in hits[0]


def test_answer_uses_library(tmp_path: Path):
    (tmp_path / "a.md").write_text("# 报销\n差旅报销单超过 5000元 需要总监审批。")
    client = FakeClient()
    out = answer_question(client, "差旅报销单超过多少元需要总监审批？", tmp_path)
    assert "5000元" in out


def test_llm_experiment_end_to_end(tmp_path: Path):
    reg = Registry.init(tmp_path / "repo")
    client = FakeClient()
    m = run_llm_experiment(reg, LLMExperimentConfig(
        benchmark=BENCH, model="fake-model", arm="ungoverned",
        rounds=2, train_per_round=6, seed=1,
    ), client=client)

    assert len(m["rounds"]) == 2
    # 第二轮技能库已积累全部 12 条事实 → eval 应显著好于第一轮
    assert m["rounds"][1]["eval_accuracy"] >= m["rounds"][0]["eval_accuracy"]
    assert m["cost"]["calls"] > 0
    manifest = reg.root / "runs" / m["run_id"] / "manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["config"]["model"] == "fake-model"
    assert (reg.root / "runs" / m["run_id"] / "trajectories_r1.json").exists()
