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


def cmd_gate(args) -> int:
    from skillops.gate.engine import Gate

    reg = _open(args)
    gate = Gate(benchmark=args.bench, min_delta=args.min_delta)
    verdict = gate.evaluate(reg, args.vid)

    layers = verdict.metrics.get("layers", {})
    static = layers.get("static", {})
    print(f"[static]   {'PASS' if static.get('passed') else 'FAIL'}"
          f"  ({len(static.get('findings', []))} findings)")
    for f in static.get("findings", []):
        print(f"           {f['severity']:5s} {f['check']}: {f['message']}")
    for name in ("quick", "full", "holdout"):
        if name in layers:
            L = layers[name]
            n = L["tasks"] if isinstance(L["tasks"], int) else len(L["tasks"])
            print(f"[{name:8s}] baseline {L['baseline']:.3f} → candidate {L['candidate']:.3f}"
                  f"  (delta {L['delta']:+.3f}, {n} tasks)")
    print(f"\nverdict: {'PASS' if verdict.passed else 'FAIL'} — {verdict.reason}")
    return 0 if verdict.passed else 1


def _not_in_m1(name: str):
    def _cmd(args) -> int:
        print(f"`skillops {name}` {_M2_HINT}")
        return 2
    return _cmd


def cmd_experiment_run(args) -> int:
    reg = _open(args)

    if args.evolver == "llm":
        from skillops.experiments.llm_pipeline import (
            LLMExperimentConfig,
            run_llm_experiment,
        )
        cfg = LLMExperimentConfig(
            benchmark=args.bench,
            model=args.model,
            arm=args.arm,
            rounds=args.rounds,
            seed=args.seed,
            min_delta=args.min_delta,
            out=args.out,
            eval_max_tasks=args.eval_max_tasks,
            max_workers=args.max_workers,
        )
        m = run_llm_experiment(reg, cfg)
    else:
        from skillops.experiments.runner import ExperimentConfig, run_experiment
        cfg = ExperimentConfig(
            benchmark=args.bench,
            evolver=args.evolver,
            arm=args.arm,
            rounds=args.rounds,
            seed=args.seed,
            min_delta=args.min_delta,
            out=args.out,
        )
        m = run_experiment(reg, cfg)

    print(f"run_id: {m['run_id']}  (arm={cfg.arm}, evolver={args.evolver}, seed={cfg.seed})")
    print(f"{'rnd':>3}  {'decision':10}  {'eval_acc':>8}  {'lib_size':>8}  vid")
    for r in m["rounds"]:
        print(f"{r['round']:>3}  {r['decision']:10}  {r['eval_accuracy']:>8.3f}"
              f"  {r['library_size']:>8}  {r['vid']}")
    f = m["final"]
    print(f"\nfinal: eval {f['eval_accuracy']:.3f}  peak {f['peak_eval_accuracy']:.3f}"
          f"  holdout {f['holdout_accuracy']:.3f}  lib {f['library_size']}")
    cost = m.get("cost", {})
    if cost.get("total_tokens"):
        print(f"cost: {cost['calls']} API calls, "
              f"prompt {cost['prompt_tokens']} + completion {cost['completion_tokens']}"
              f" = {cost['total_tokens']} tokens")
    print(f"manifest: {cfg.out or reg.root / 'runs'}/{m['run_id']}/manifest.json")
    return 0


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

    sp = sub.add_parser("gate", help="run layered evaluation gate on a staged version")
    sp.add_argument("vid")
    sp.add_argument("--bench", required=True, help="path to a benchmark directory")
    sp.add_argument("--min-delta", type=float, default=0.0,
                    help="minimum accuracy delta vs baseline (default 0.0 = no regression)")
    sp.add_argument("--repo", default=".")
    sp.set_defaults(func=cmd_gate)

    for name in ("sweep", "scan", "rollback", "report"):
        sp = sub.add_parser(name, help=f"(M2) {_M2_HINT}")
        sp.set_defaults(func=_not_in_m1(name))

    sp = sub.add_parser("experiment", help="run evolution experiments (RQ1/RQ2)")
    exp_sub = sp.add_subparsers(dest="exp_command", required=True)
    run_p = exp_sub.add_parser("run", help="run one evolution experiment")
    run_p.add_argument("--bench", required=True, help="path to a benchmark directory")
    run_p.add_argument("--evolver", default="scripted",
                       choices=["scripted", "llm"],
                       help="llm = DeepSeek 真实进化环（需要 DEEPSEEK_API_KEY）")
    run_p.add_argument("--model", default="deepseek-chat",
                       help="LLM evolver/agent 使用的模型（仅 --evolver llm）")
    run_p.add_argument("--arm", default="ungoverned", choices=["ungoverned", "gate"],
                       help="ungoverned = commit everything; gate = 3-layer gate")
    run_p.add_argument("--rounds", type=int, default=10)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--min-delta", type=float, default=0.0)
    run_p.add_argument("--out", default=None, help="runs output dir (default: <repo>/runs)")
    run_p.add_argument("--eval-max-tasks", type=int, default=0,
                       help="截断 eval split（smoke 成本校准用，0 = 不截断）")
    run_p.add_argument("--max-workers", type=int, default=6, help="API 并发度")
    run_p.add_argument("--repo", default=".")
    run_p.set_defaults(func=cmd_experiment_run)

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
