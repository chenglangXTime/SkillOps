"""skillops CLI — M1 scope: init / stage / status / commit / reject / log / discover."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skillops.core.models import SkillStatus
from skillops.core.registry import Registry, RegistryError
from skillops.adapters.claude_code import discover_skills

_M2_HINT = "ships in M2 — see SkillOps开源项目提案.md §6"


def _open(args) -> Registry:
    return Registry(getattr(args, "repo", "."))


def cmd_init(args) -> int:
    reg = Registry.init(args.path)
    print(f"Initialized skillops repo at {reg.root}")
    print(f"  skills/        — 技能工件目录")
    print(f"  trajectories/  — 来源轨迹存档")
    print(f"  .skillops/     — 账本与配置")
    return 0


def cmd_stage(args) -> int:
    reg = _open(args)
    v = reg.stage(
        args.skill_file,
        source_trajectories=args.source or [],
        generator=args.generator or "",
    )
    print(f"staged {v.vid}  (sha256={v.sha256[:12]}…, status={v.status.value})")
    if v.parent_ids:
        print(f"  lineage: {' ← '.join(v.parent_ids)} → {v.vid}")
    return 0


def cmd_status(args) -> int:
    reg = _open(args)
    s = reg.status()
    print(f"repo: {s['root']}  (git HEAD: {s['git_head']})")
    print(f"versions: {s['total_versions']}  {s['by_status']}")
    for v in reg.list_versions():
        print(f"  [{v.status.value:9s}] {v.vid:30s} {v.name}")
    return 0


def cmd_commit(args) -> int:
    reg = _open(args)
    commit_hash = reg.commit(args.vid)
    print(f"committed {args.vid}  (git: {commit_hash})")
    return 0


def cmd_reject(args) -> int:
    reg = _open(args)
    reg.reject(args.vid, reason=args.reason or "")
    print(f"rejected {args.vid}")
    return 0


def cmd_log(args) -> int:
    reg = _open(args)
    for e in reg.log(limit=args.limit):
        print(f"{e['ts']}  {e['action']:8s}  {e['detail']}")
    return 0


def cmd_discover(args) -> int:
    paths = discover_skills(args.path)
    for p in paths:
        print(p)
    print(f"\n{len(paths)} SKILL.md file(s) found under {Path(args.path).resolve()}")
    return 0


def _not_in_m1(name: str):
    def _cmd(args) -> int:
        print(f"`skillops {name}` {_M2_HINT}")
        return 2
    return _cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillops",
        description="Lifecycle governance for self-evolving agent skill libraries.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="initialize a skill repo")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("stage", help="register a candidate skill (staging area)")
    sp.add_argument("skill_file")
    sp.add_argument("--from", dest="source", nargs="*", metavar="TRAJ",
                    help="source trajectory ids")
    sp.add_argument("--generator", help="evolver that produced this skill")
    sp.add_argument("--repo", default=".")
    sp.set_defaults(func=cmd_stage)

    sp = sub.add_parser("status", help="show repo status and versions")
    sp.add_argument("--repo", default=".")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("commit", help="promote a staged version into the library")
    sp.add_argument("vid")
    sp.add_argument("--repo", default=".")
    sp.set_defaults(func=cmd_commit)

    sp = sub.add_parser("reject", help="reject a staged version")
    sp.add_argument("vid")
    sp.add_argument("--reason", default="")
    sp.add_argument("--repo", default=".")
    sp.set_defaults(func=cmd_reject)

    sp = sub.add_parser("log", help="show audit log")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--repo", default=".")
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser("discover", help="find SKILL.md files under a directory")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_discover)

    for name in ("gate", "sweep", "scan", "rollback", "report"):
        sp = sub.add_parser(name, help=f"(M2) {_M2_HINT}")
        sp.set_defaults(func=_not_in_m1(name))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
