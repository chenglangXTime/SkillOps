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
    harness: str                       # command template
    tasks: list[dict] = field(default_factory=list)
    quick_sample: list[str] = field(default_factory=list)
    holdout: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Benchmark":
        root = Path(path).resolve()
        cfg_path = root / CONFIG_NAME
        if not cfg_path.is_file():
            raise BenchmarkError(f"benchmark config not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        tasks = []
        tasks_file = root / cfg.get("tasks_file", "tasks.jsonl")
        if tasks_file.is_file():
            for line in tasks_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    tasks.append(json.loads(line))

        return cls(
            name=cfg.get("name", root.name),
            root=root,
            harness=cfg["harness"],
            tasks=tasks,
            quick_sample=cfg.get("quick_sample", []),
            holdout=cfg.get("holdout", []),
            description=cfg.get("description", ""),
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
        """Run the harness against a library view; return parsed JSON result."""
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
