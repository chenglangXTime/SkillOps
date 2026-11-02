"""officeqa_mini harness — deterministic, dependency-free, retrieval-based.

Simulates a retrieval-augmented agent:

- The model's base knowledge is the BASE_KNOWLEDGE constant.
- The skill library is every file under --library.
- For each question, the harness retrieves the single best-matching document
  (character-bigram overlap; base knowledge included as one document) and
  answers from THAT document only.

Retrieval-based scoring makes library noise measurable: a degenerate skill
with high keyword overlap but an outdated answer displaces the correct skill
and lowers accuracy — the canonical "unmanaged library drift" failure mode.
Fully reproducible, zero LLM calls.

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


def _bigrams(text: str) -> set[tuple[str, str]]:
    chars = [c for c in text.lower() if c.isalnum()]
    return set(zip(chars, chars[1:]))


def _retrieve(question: str, docs: list[tuple[str, str]]) -> str:
    """Return the content of the best-matching doc; ties prefer earlier docs
    (BASE is always first in the list)."""
    q = _bigrams(question)
    best_content, best_score = docs[0][1], len(q & _bigrams(docs[0][1]))
    for _, content in docs[1:]:
        score = len(q & _bigrams(content))
        if score > best_score:
            best_content, best_score = content, score
    return best_content


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

    docs: list[tuple[str, str]] = [("BASE", BASE_KNOWLEDGE)]
    if args.library.is_dir():
        for f in sorted(args.library.iterdir()):
            if f.is_file():
                docs.append((f.name, f.read_text(encoding="utf-8", errors="ignore")))

    scores = {}
    for t in tasks:
        doc = _retrieve(t["question"], docs)
        scores[t["id"]] = int(t["answer"].lower() in doc.lower())

    n = len(tasks)
    result = {
        "scores": scores,
        "accuracy": (sum(scores.values()) / n) if n else 0.0,
        "n": n,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
