"""Smoke test: full M1 flow init → stage → commit → supersede → audit."""

from pathlib import Path

from skillops.core.models import SkillStatus
from skillops.core.registry import Registry


def test_full_m1_flow(tmp_path: Path):
    repo = tmp_path / "repo"
    reg = Registry.init(repo)
    assert (repo / ".skillops" / "ledger.db").exists()
    assert (repo / ".git").exists()

    skill = tmp_path / "refund_triage.md"
    skill.write_text("# Refund Triage\nv1 body\n")

    v1 = reg.stage(skill, source_trajectories=["traj_001"], generator="evoskill-v0.3")
    assert v1.vid == "refund_triage@v1"
    assert v1.status is SkillStatus.STAGED

    reg.commit(v1.vid, metrics={"officeqa_mini": 0.61})
    assert reg.ledger.get(v1.vid).status is SkillStatus.COMMITTED

    # v2 supersedes v1
    skill.write_text("# Refund Triage\nv2 body\n")
    v2 = reg.stage(skill)
    assert v2.vid == "refund_triage@v2"
    assert v2.parent_ids == [v1.vid]

    reg.commit(v2.vid)
    assert reg.ledger.get(v1.vid).status is SkillStatus.RETIRED
    assert reg.ledger.get(v2.vid).status is SkillStatus.COMMITTED

    actions = [e["action"] for e in reg.log()]
    assert actions[:1] == ["commit"] or "commit" in actions
    assert "init" in actions and "stage" in actions and "retire" in actions
