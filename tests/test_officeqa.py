"""OfficeQA integration tests: prepare/split, corpus retrieval, fuzzy judging."""

import json
import sys
from pathlib import Path

import pytest

from skillops.gate.benchmark import Benchmark
from skillops.experiments.llm_pipeline import (
    make_judge,
    make_llm_eval_fn,
    retrieve_corpus_chunks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "officeqa"))
import prepare  # noqa: E402


def _rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "uid": f"qid_{i}",
            "question": f"Question {i}?",
            "answer": str(1000 + i),
            "source_files": f"doc_{i % 5}.txt",
            "difficulty": "hard" if i % 2 == 0 else "easy",
        })
    return rows


def test_split_deterministic_and_stratified():
    rows = _rows(20)
    tr1, va1, ho1 = prepare.split_rows(rows, 8, 4, seed=7)
    tr2, va2, ho2 = prepare.split_rows(rows, 8, 4, seed=7)
    assert [r["uid"] for r in tr1] == [r["uid"] for r in tr2]     # 确定性
    assert len(tr1) + len(va1) + len(ho1) == 20
    assert len(tr1) == 8 and len(va1) == 4
    ids = {r["uid"] for r in tr1 + va1 + ho1}
    assert len(ids) == 20                                          # 无重叠
    tr3, _, _ = prepare.split_rows(rows, 8, 4, seed=8)
    assert [r["uid"] for r in tr3] != [r["uid"] for r in tr1]      # seed 生效


def test_write_and_load_benchmark(tmp_path: Path):
    rows = _rows(20)
    tr, va, ho = prepare.split_rows(rows, 8, 4, seed=1)
    prepare.write_benchmark(tmp_path, tr, va, ho)

    bench = Benchmark.load(tmp_path)
    assert bench.scoring == "officeqa_fuzzy"
    assert bench.corpus_dir == "corpus"
    assert len(bench.train_tasks) == 8
    assert len(bench.eval_split()) == 4           # val = 门控/监控集
    assert len(bench.holdout_split()) == 8        # 其余 = 锁定集
    assert all(t["split"] == "val" for t in bench.tasks
               if t["id"] in set(bench.eval_split()))
    # 引用文件清单
    files = prepare.referenced_files(tmp_path)
    assert files == [f"doc_{i}.txt" for i in range(5)]


def test_corpus_chunk_retrieval(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_a.txt").write_text(
        "第一段：无关内容。\n\n"
        "第二段：1941 年 1 月国债发行总额为 $7,046,001.98，详见表 3。\n\n"
        "第三段：其他无关数据。" * 3)
    (corpus / "doc_b.txt").write_text("完全无关的文档。" * 50)

    chunks = retrieve_corpus_chunks(
        "1941 年 1 月国债发行总额是多少？",
        corpus, ["doc_a.txt", "doc_b.txt"], chunk_chars=200, top_chunks=2)
    assert chunks and any("7,046,001.98" in c for c in chunks)


def test_fuzzy_judge_dispatch(tmp_path: Path):
    rows = _rows(4)
    tr, va, ho = prepare.split_rows(rows, 2, 1, seed=1)
    prepare.write_benchmark(tmp_path, tr, va, ho)
    # 移动 vendor 判分器到测试基准目录
    vendor_src = Path(__file__).resolve().parent.parent / \
        "benchmarks" / "officeqa" / "vendor"
    (tmp_path / "vendor").mkdir(exist_ok=True)
    (tmp_path / "vendor" / "reward.py").write_text(
        (vendor_src / "reward.py").read_text(encoding="utf-8"))

    bench = Benchmark.load(tmp_path)
    fuzzy = make_judge(bench)
    assert fuzzy("5000", "推理…… <FINAL_ANSWER>5000</FINAL_ANSWER>") == 1
    assert fuzzy("$7,046,001.98", "<FINAL_ANSWER>7046001.98</FINAL_ANSWER>") == 1
    assert fuzzy("5000", "<FINAL_ANSWER>3000</FINAL_ANSWER>") == 0


class _CorpusFakeClient:
    model = "fake"
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, max_tokens=1024, temperature=None):
        self.usage["calls"] += 1
        ctx = messages[-1]["content"]
        for marker in ("$1,234", "$9,999"):
            if marker in ctx:
                return f"依据文档…… <FINAL_ANSWER>{marker}</FINAL_ANSWER>"
        return "<FINAL_ANSWER>unknown</FINAL_ANSWER>"


def test_corpus_mode_eval_fn(tmp_path: Path):
    # 手工构造一个语料型基准
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "doc_a.txt").write_text(
        "表 1：1939 年财政总收入为 $1,234。\n\n" + "填充段落。" * 20)
    (tmp_path / "tasks.jsonl").write_text(json.dumps({
        "id": "v1", "question": "1939 年财政总收入是多少？", "answer": "$1,234",
        "source_files": ["doc_a.txt"], "split": "val",
    }, ensure_ascii=False) + "\n")
    (tmp_path / "train_tasks.jsonl").write_text(json.dumps({
        "id": "t1", "question": "某年支出是多少？", "answer": "$9,999",
        "source_files": ["doc_a.txt"], "split": "train",
    }, ensure_ascii=False) + "\n")
    vendor_src = Path(__file__).resolve().parent.parent / \
        "benchmarks" / "officeqa" / "vendor"
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "reward.py").write_text(
        (vendor_src / "reward.py").read_text(encoding="utf-8"))
    (tmp_path / "benchmark.json").write_text(json.dumps({
        "name": "toy_corpus", "tasks_file": "tasks.jsonl",
        "train_file": "train_tasks.jsonl", "quick_sample": ["v1"], "holdout": [],
        "scoring": "officeqa_fuzzy", "corpus_dir": "corpus",
        "chunk_chars": 200, "top_chunks": 3,
    }, ensure_ascii=False))

    bench = Benchmark.load(tmp_path)
    eval_fn = make_llm_eval_fn(_CorpusFakeClient(), bench)
    res = eval_fn(tmp_path, ["v1"])
    assert res["accuracy"] == pytest.approx(1.0)   # 检索到语料块 → 答对
