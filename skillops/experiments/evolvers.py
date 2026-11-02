"""Evolvers: pluggable skill-proposal sources for experiments.

- `ScriptedEvolver`: 确定性、零 LLM。在 officeqa_mini 域上模拟一条真实的
  技能进化轨迹——先产真正有用的政策技能，随后产出退化的"经验"
  （过期数值、高关键词重合的近似重复），通过检索干扰拉低库整体准确率。
  用途：验证测量装置（RQ1 可行性）；论文正式实验用 `LLMEvolver`。
- `LLMEvolver`: 论文用占位——对接任意 OpenAI 兼容 API，从真实轨迹蒸馏技能。
"""

from __future__ import annotations

import random
from typing import Protocol

# 与 officeqa_mini q07-q12 对应的正确政策池
_CORRECT_POLICIES = [
    ("expense", "差旅报销审批政策：差旅报销单超过 5000元 需要总监审批。"),
    ("leave", "请假政策：病假超过 3天 需要提供三甲医院证明。"),
    ("meeting", "会议室政策：会议室预订系统提前 14天 开放预订。"),
    ("travel", "差旅交通标准：高铁二等座差旅标准最高单价 800元。"),
    ("it", "IT 支持指南：VPN 故障请拨打 IT 支持热线分机 8848。"),
    ("procurement", "采购政策：单笔采购超过 50000元 必须走招标流程。"),
]

# 退化模板 A：与技能问题高关键词重合，但给出过期/错误数值（检索干扰）
_OUTDATED_SKILL_DECOYS = [
    ("expense_old", "差旅报销单 超过 多少元 需要 总监 审批："
     "旧制度下差旅报销单超过 3000元 需要总监审批。本条为历史工单沉淀的审批经验。"),
    ("leave_old", "病假 超过 多少天 需要 三甲医院 证明："
     "旧制度下病假超过 7天 才需要三甲医院证明。本条为历史请假记录提炼的经验。"),
    ("meeting_old", "会议室预订系统 提前 多少天 开放预订："
     "旧系统会议室预订提前 3天 开放。本条为旧版行政手册中的会议室预订经验。"),
    ("travel_old", "高铁二等座差旅标准的最高单价是多少："
     "旧标准下高铁二等座差旅最高单价为 600元。本条为往年差旅报销总结的经验。"),
    ("it_old", "VPN 故障时 IT 支持热线的分机号是多少："
     "旧办公区 IT 支持热线分机号为 6633。本条为搬家前 IT 工单沉淀的经验。"),
    ("procurement_old", "单笔采购超过多少元必须走招标流程："
     "旧制度下单笔采购超过 20000元 即须招标。本条为历史采购记录提炼的经验。"),
]

# 退化模板 B：侵蚀基础知识题（把 BASE 的正确答案挤下检索首位）
_OUTDATED_BASE_DECOYS = [
    ("salary_old", "每月几号发放工资的经验谈：旧制度下每月 15号 发放工资，"
     "遇节假日顺延。本条为多年前薪酬记录提炼的经验。"),
    ("worktime_old", "公司标准工作时间是几点到几点的经验：旧办公区标准工作时间为 "
     "9:00-18:00。本条为搬迁前考勤制度沉淀的经验。"),
]

_FILLER = "本条经验由系统自动从历史记录提炼，供后续任务检索参考。"


class Evolver(Protocol):
    name: str

    def propose(self, round_idx: int) -> tuple[str, str]:
        """Return (skill_name, markdown_content) for this round."""
        ...


class ScriptedEvolver:
    """Deterministic evolution trajectory: useful skills first, then drift."""

    name = "scripted"

    def __init__(self, seed: int = 42, n_useful: int = 3):
        self.rng = random.Random(seed)
        self.n_useful = n_useful
        self._policies = list(_CORRECT_POLICIES)
        self.rng.shuffle(self._policies)
        self._decoys = list(_OUTDATED_SKILL_DECOYS) + list(_OUTDATED_BASE_DECOYS)
        self.rng.shuffle(self._decoys)
        self._decoy_idx = 0

    def propose(self, round_idx: int) -> tuple[str, str]:
        if round_idx <= self.n_useful:
            batch = self._policies[
                (round_idx - 1) * 2 : (round_idx - 1) * 2 + 2
            ] or self._policies[:1]
            lines = "\n".join(f"- {text}" for _, text in batch)
            content = (
                f"# 政策汇编技能（第 {round_idx} 批）\n\n"
                f"本技能从近期工单中蒸馏，覆盖以下现行政策：\n\n{lines}\n\n"
                f"{_FILLER}"
            )
            return f"policy_pack_r{round_idx}", content

        key, decoy = self._decoys[self._decoy_idx % len(self._decoys)]
        self._decoy_idx += 1
        content = (
            f"# 历史经验：{key}\n\n{decoy}\n\n{_FILLER}"
        )
        return f"{key}_r{round_idx}", content


class LLMEvolver:  # pragma: no cover - paper runs
    """LLM-backed evolver for the paper's real experiments.

    将真实执行轨迹蒸馏为候选技能（对接任意 OpenAI 兼容 API）。
    成本（token 数）会通过 RunManifest.cost 落账。M3 实现。
    """

    name = "llm"

    def __init__(self, model: str, seed: int = 42, **kwargs):
        raise NotImplementedError(
            "LLMEvolver ships in M3 — it is the paper-run evolver; "
            "use ScriptedEvolver for apparatus validation."
        )


def make_evolver(name: str, seed: int = 42, **kwargs) -> Evolver:
    if name == "scripted":
        return ScriptedEvolver(seed=seed, **kwargs)
    if name == "llm":
        return LLMEvolver(seed=seed, **kwargs)
    raise ValueError(f"unknown evolver: {name}")
