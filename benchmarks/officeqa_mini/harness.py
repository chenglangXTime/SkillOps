"""officeqa_mini harness — deterministic, dependency-free.

Simulates a retrieval-augmented agent: the model's base knowledge is the
BASE_KNOWLEDGE constant; the skill library is every file under --library.
A task is scored correct if its gold answer string appears in the combined
text. This makes skill contributions measurable end-to-end (adding the right
skill raises accuracy) while staying fully reproducible with zero LLM calls.

Usage:
    python3 harness.py --library <dir> --tasks q01,q07 --output result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_KNOWLEDGE = """
员工手册通用知识：
公司标准工作时间为 9:30-18:30，弹性一小时。
工资于每月 10号 发放，遇节假日提前。
入职体检报销上限 300元。
总部大会议厅最大容纳 120人。
工卡遗失补办费用 50元，三个工作日内领取。
员工法定年假最少 5天，随司龄递增。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True, type=Path)
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    bench_dir = Path(__file__).resolve().parent
    wanted = set(args.tasks.split(","))
    tasks = []
    for line in (bench_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            if t["id"] in wanted:
                tasks.append(t)

    corpus = BASE_KNOWLEDGE
    if args.library.is_dir():
        for f in sorted(args.library.iterdir()):
            if f.is_file():
                corpus += "\n" + f.read_text(encoding="utf-8", errors="ignore")
    corpus = corpus.lower()

    scores = {t["id"]: int(t["answer"].lower() in corpus) for t in tasks}
    n = len(tasks)
    result = {
        "scores": scores,
        "accuracy": (sum(scores.values()) / n) if n else 0.0,
        "n": n,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
