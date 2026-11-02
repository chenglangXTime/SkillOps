"""Benchmark loading and harness execution.

一个 SkillOps 基准就是一个目录：

    benchmarks/officeqa_mini/
    ├── benchmark.json   # 元信息 + harness 命令模板 + quick/holdout 划分
    ├── tasks.jsonl      # 任务集（每行 {"id", "question", "answer", ...}）
    └── harness.py       # 执行器：给定技能库目录与任务子集，输出 JSON 分数

harness 命令模板支持占位符：{python} {benchmark_dir} {library_dir} {task_ids} {output_json}
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "benchmark.json"


class BenchmarkError(RuntimeError):
    pass


@dataclass
class Benchmark:
    name: str
    root: Path
    harness: str = ""                  # command template; empty for LLM-driven benchmarks
    tasks: list[dict] = field(default_factory=list)
    train_tasks: list[dict] = field(default_factory=list)
    quick_sample: list[str] = field(default_factory=list)
    holdout: list[str] = field(default_factory=list)
    description: str = ""
    scoring: str = "substring"         # "substring" | "officeqa_fuzzy"
    tolerance: float = 0.0
    corpus_dir: str = ""               # 非空 → 语料检索模式（OfficeQA）
    chunk_chars: int = 4000
    top_chunks: int = 6

    @classmethod
    def load(cls, path: str | Path) -> "Benchmark":
        root = Path(path).resolve()
        cfg_path = root / CONFIG_NAME
        if not cfg_path.is_file():
            raise BenchmarkError(f"benchmark config not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        def _load_jsonl(fname: str | None) -> list[dict]:
            if not fname:
                return []
            f = root / fname
            if not f.is_file():
                return []
            return [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines()
                    if ln.strip()]

        return cls(
            name=cfg.get("name", root.name),
            root=root,
            harness=cfg.get("harness", ""),
            tasks=_load_jsonl(cfg.get("tasks_file", "tasks.jsonl")),
            train_tasks=_load_jsonl(cfg.get("train_file")),
            quick_sample=cfg.get("quick_sample", []),
            holdout=cfg.get("holdout", []),
            description=cfg.get("description", ""),
            scoring=cfg.get("scoring", "substring"),
            tolerance=float(cfg.get("tolerance", 0.0)),
            corpus_dir=cfg.get("corpus_dir", ""),
            chunk_chars=int(cfg.get("chunk_chars", 4000)),
            top_chunks=int(cfg.get("top_chunks", 6)),
        )

    # ---- task splits ----

    def eval_split(self) -> list[str]:
        """Full evaluation split: everything except the locked holdout."""
        hold = set(self.holdout)
        return [t["id"] for t in self.tasks if t["id"] not in hold]

    def quick_split(self) -> list[str]:
        if self.quick_sample:
            return list(self.quick_sample)
        return self.eval_split()[:4]

    def holdout_split(self) -> list[str]:
        return list(self.holdout)

    # ---- execution ----

    def run(self, library_dir: Path, task_ids: list[str]) -> dict:
        """Run the subprocess harness against a library view (deterministic benchmarks).

        LLM-driven benchmarks（如 OfficeQA，harness 为空）请通过
        `make_llm_eval_fn` 注入评估器，不走此路径。
        """
        if not self.harness:
            raise BenchmarkError(
                f"benchmark {self.name} has no subprocess harness; "
                "use an LLM eval_fn (see experiments.llm_pipeline)"
            )
        if not task_ids:
            return {"scores": {}, "accuracy": 0.0, "n": 0}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as out:
            output_json = Path(out.name)

        cmd = self.harness.format(
            python=sys.executable,
            benchmark_dir=self.root,
            library_dir=library_dir,
            task_ids=",".join(task_ids),
            output_json=output_json,
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BenchmarkError(
                f"harness failed ({proc.returncode}): {proc.stderr.strip()[:500]}"
            )
        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        finally:
            output_json.unlink(missing_ok=True)
