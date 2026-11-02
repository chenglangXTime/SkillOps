"""Gate flow tests: layered evaluation on the bundled officeqa_mini benchmark."""

from pathlib import Path

import pytest

from skillops.core.registry import Registry, RegistryError
from skillops.gate.engine import Gate

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "officeqa_mini"

GOOD_SKILL = """# 办公室政策速查技能

用于回答报销、请假、会议室、差旅、IT 与采购相关问题。

- 差旅报销单超过 5000元 需要总监审批。
- 病假超过 3天 需要提供三甲医院证明。
- 会议室预订系统提前 14天 开放预订。
- 高铁二等座差旅标准最高单价 800元。
- VPN 故障请拨打 IT 支持热线分机 8848。
- 单笔采购超过 50000元 必须走招标流程。
"""

EMPTY_SKILL = "# 无关技能\n\n这份技能记录的是完全无关的运维笔记，不包含任何办公室政策答案，" + "凑够长度。" * 20

LEAKY_SKILL = "# 泄露密钥的技能\n\napi_key = 'sk-abcdef0123456789abcdef0123'\n这段技能包含硬编码凭据，应被静态检查拦截。" + "补充说明文字。" * 10


@pytest.fixture()
def reg(tmp_path: Path) -> Registry:
    return Registry.init(tmp_path / "repo")


def test_gate_passes_with_useful_skill(reg: Registry, tmp_path: Path):
    skill = tmp_path / "office_policy.md"
    skill.write_text(GOOD_SKILL)
    v = reg.stage(skill, generator="test")

    verdict = Gate(BENCH).evaluate(reg, v.vid)

    assert verdict.passed, verdict.reason
    layers = verdict.metrics["layers"]
    assert layers["static"]["passed"]
    # baseline 只能答 base 题；candidate 全对
    assert layers["full"]["delta"] == pytest.approx(3 / 9)
    assert layers["holdout"]["delta"] == pytest.approx(1.0)
    # 指标已写入账本
    assert "gate" in reg.ledger.get(v.vid).metrics


def test_gate_fails_without_improvement(reg: Registry, tmp_path: Path):
    skill = tmp_path / "irrelevant.md"
    skill.write_text(EMPTY_SKILL)
    v = reg.stage(skill)

    verdict = Gate(BENCH, min_delta=0.05).evaluate(reg, v.vid)

    assert not verdict.passed
    assert verdict.metrics["layers"]["full"]["delta"] == pytest.approx(0.0)


def test_gate_blocks_leaked_secret(reg: Registry, tmp_path: Path):
    skill = tmp_path / "leaky.md"
    skill.write_text(LEAKY_SKILL)
    v = reg.stage(skill)

    verdict = Gate(BENCH).evaluate(reg, v.vid)

    assert not verdict.passed
    assert "static" in verdict.reason
    # fail-fast：不应进入 quick/full 层
    assert "quick" not in verdict.metrics["layers"]


def test_gate_rejects_non_staged(reg: Registry, tmp_path: Path):
    skill = tmp_path / "office_policy.md"
    skill.write_text(GOOD_SKILL)
    v = reg.stage(skill)
    reg.commit(v.vid)

    with pytest.raises(RegistryError):
        Gate(BENCH).evaluate(reg, v.vid)
