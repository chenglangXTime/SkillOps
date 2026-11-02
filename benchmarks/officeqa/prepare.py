"""prepare.py — convert gated OfficeQA data into a SkillOps benchmark directory.

用法（数据获批后）：

    # 1) 从 HF 下载 CSV（或网页下载后指定本地路径）
    python prepare.py --csv /path/to/officeqa_full.csv --out .

    # 2) 只下载题目引用到的解析 txt（而非全部 460MB）
    python prepare.py --csv officeqa_full.csv --out . --download

网络受限时设镜像：HF_ENDPOINT=https://hf-mirror.com

分割（沿用 WikiSkill / envharness 惯例）：train 50 / val 24 / test 其余。
- train → train_tasks.jsonl（evolver 反馈来源，答案只用于训练期反馈）
- val   → tasks.jsonl 中非 holdout 部分（门控准入 + 逐轮监控）
- test  → holdout（锁定，实验结束评一次）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DEFAULT_TRAIN = 50
DEFAULT_VAL = 24
QUICK_SAMPLE = 6


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_rows(rows: list[dict], n_train: int, n_val: int,
               seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic difficulty-stratified split: train / val / holdout."""
    rng = random.Random(seed)
    by_diff: dict[str, list[dict]] = {}
    for r in rows:
        by_diff.setdefault(r.get("difficulty", "hard"), []).append(r)
    for group in by_diff.values():
        rng.shuffle(group)

    train, val, hold = [], [], []
    groups = sorted(by_diff.values(), key=len, reverse=True)
    for i, g in enumerate(groups):
        t = round(n_train * len(g) / len(rows)) if len(groups) > 1 else n_train
        v = round(n_val * len(g) / len(rows)) if len(groups) > 1 else n_val
        train += g[:t]
        val += g[t:t + v]
        hold += g[t + v:]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(hold)
    return train, val, hold


def to_task(r: dict, split: str) -> dict:
    import re
    # source_files 分隔符不统一：部分行用 ";"，部分用内嵌换行（\r\n）
    raw = r.get("source_files", "")
    files = [s.strip() for s in re.split(r"[;\r\n]+", raw) if s.strip()]
    return {
        "id": r["uid"],
        "question": r["question"],
        "answer": r["answer"],
        "difficulty": r.get("difficulty", "hard"),
        "source_files": files,
        "split": split,
    }


def write_benchmark(out: Path, train: list, val: list, hold: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(exist_ok=True)

    tasks = [to_task(r, "val") for r in val] + [to_task(r, "holdout") for r in hold]
    train_tasks = [to_task(r, "train") for r in train]

    (out / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8")
    (out / "train_tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in train_tasks) + "\n",
        encoding="utf-8")

    cfg = {
        "name": "officeqa",
        "version": "1.0",
        "description": "Databricks OfficeQA (gated, CC-BY-SA 4.0). val=gate/monitor, "
                       "test=locked holdout. Fuzzy scoring via vendored reward.py.",
        "tasks_file": "tasks.jsonl",
        "train_file": "train_tasks.jsonl",
        "quick_sample": [t["id"] for t in tasks[:QUICK_SAMPLE] if t["split"] == "val"],
        "holdout": [t["id"] for t in tasks if t["split"] == "holdout"],
        "scoring": "officeqa_fuzzy",
        "tolerance": 0.0,
        "corpus_dir": "corpus",
        "chunk_chars": 4000,
        "top_chunks": 6,
    }
    (out / "benchmark.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared: train={len(train)} val={len(val)} holdout={len(hold)}")
    print(f"tasks -> {out}/tasks.jsonl, train -> {out}/train_tasks.jsonl")


def referenced_files(out: Path) -> list[str]:
    files = set()
    for fname in ("tasks.jsonl", "train_tasks.jsonl"):
        p = out / fname
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    files.update(json.loads(line).get("source_files", []))
    return sorted(files)


def _load_hf_token() -> str | None:
    import os
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = Path.home() / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


def download_corpus(out: Path, repo_id: str = "databricks/officeqa") -> None:
    """Download only the parsed txts referenced by the prepared tasks.

    纯 urllib 实现（零额外依赖）；token 从环境变量或 ~/.env 读取。
    网络受限时设 HF_ENDPOINT=https://hf-mirror.com。
    """
    import os
    import urllib.request

    token = _load_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found (env or ~/.env)")
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    files = referenced_files(out)
    corpus = out / "corpus"
    corpus.mkdir(exist_ok=True)
    for i, name in enumerate(files, 1):
        dest = corpus / name
        if dest.exists():
            continue
        url = (f"{base}/datasets/{repo_id}/resolve/main/"
               f"treasury_bulletins_parsed/transformed/{name}")
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
        if i % 20 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] downloaded")
    print(f"corpus ready: {len(list(corpus.glob('*.txt')))} files in {corpus}")


def fetch_csv(dest: Path, filename: str = "officeqa_full.csv",
              repo_id: str = "databricks/officeqa") -> Path:
    """Download a benchmark CSV (gated) to `dest`."""
    import os
    import urllib.request

    token = _load_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found (env or ~/.env)")
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = f"{base}/datasets/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.read())
    print(f"fetched {filename} -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, help="officeqa_full.csv（本地路径）")
    ap.add_argument("--fetch", action="store_true",
                    help="先用 HF_TOKEN 下载 CSV 到 <out>/data/（无需手动下载）")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--train", type=int, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=int, default=DEFAULT_VAL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download", action="store_true",
                    help="download referenced corpus txts (needs HF_TOKEN)")
    args = ap.parse_args()

    csv_path = args.csv
    if args.fetch or not csv_path:
        csv_path = fetch_csv(args.out / "data" / "officeqa_full.csv")

    rows = load_rows(csv_path)
    train, val, hold = split_rows(rows, args.train, args.val, args.seed)
    write_benchmark(args.out, train, val, hold)
    if args.download:
        download_corpus(args.out)


if __name__ == "__main__":
    main()
