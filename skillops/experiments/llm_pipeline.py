"""LLM experiment pipeline — real agent + real evolver (论文实验路径).

与 scripted 路径的区别：三个组件全部是真的——

- **Agent**：DeepSeek 模型回答问题，上下文 = 从当前技能库检索的 top-k 技能
- **Evolver**：DeepSeek 模型读取本轮执行轨迹（含对错反馈与正确答案），
  修订技能文档（Dynamic-Cheatsheet 式设计：训练题的正确答案只用于反馈）
- **评测**：同一 Agent 在 eval split 上作答，判分 = 标准答案字符串匹配

数据流（每轮）：
    train 任务 ×4 → Agent 作答（带当前技能库）→ 轨迹
    → LLMEvolver 修订技能 → stage → [arm: ungoverned 直接入库 / gate 三层门控]
    → eval split 评一次 → 记录
锁定纪律不变：holdout 只在实验结束时评一次。
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import skillops
from skillops.core.models import utc_now
from skillops.core.registry import Registry
from skillops.gate.benchmark import Benchmark
from skillops.gate.engine import Gate
from skillops.llm.client import DeepSeekClient


# ---------- retrieval ----------

def _bigrams(text: str) -> set[tuple[str, str]]:
    chars = [c for c in text.lower() if c.isalnum()]
    return set(zip(chars, chars[1:]))


def retrieve_topk(question: str, library_dir: Path, k: int = 2) -> list[str]:
    """Return the top-k skill texts by bigram overlap (empty overlap excluded)."""
    q = _bigrams(question)
    scored = []
    lib = Path(library_dir)
    if lib.is_dir():
        for f in sorted(lib.iterdir()):
            if f.is_file():
                content = f.read_text(encoding="utf-8", errors="ignore")
                score = len(q & _bigrams(content))
                if score > 0:
                    scored.append((score, f.name, content))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored[:k]]


# ---------- corpus retrieval (OfficeQA 等语料型基准) ----------

_CHUNK_CACHE: dict[str, list[str]] = {}

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_FILE_PERIOD_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt")
_DIGIT_RE = re.compile(r"\d")
_PHRASE_RE = re.compile(r"\b(?:[A-Z][\w'-]*(?:\s+(?:of|the|for|and|in|on)\s+|\s+|$)){2,}")
_PHRASE_STOP = {"from", "the", "a", "an", "according", "using", "considering",
                "to", "in", "on", "for", "of", "and", "what", "was", "were",
                "is", "are", "by", "with", "between", "during", "as"}


def _question_phrases(question: str) -> list[str]:
    """题目中的大写多词专名（如 'Exchange Stabilization Fund'）——强检索信号。

    句首虚词会被正则一并吞入（'From the Exchange...'），需剥掉前导虚词，
    只保留从首个大写内容词开始的连续段。
    """
    phrases = []
    for m in _PHRASE_RE.finditer(question):
        words = m.group(0).split()
        while words and (words[0].lower() in _PHRASE_STOP or not words[0][0].isupper()):
            words = words[1:]
        # 中段若出现小写非连接词，截断到该处之前
        cut = len(words)
        for i, w in enumerate(words):
            if not w[0].isupper() and w.lower() not in _PHRASE_STOP:
                cut = i
                break
        words = words[:cut]
        if len(words) >= 2:
            phrases.append(" ".join(words).lower())
    # 去重（保留子串中较长者）
    phrases.sort(key=len, reverse=True)
    kept: list[str] = []
    for p in phrases:
        if not any(p in k for k in kept):
            kept.append(p)
    return kept


def _question_periods(question: str) -> tuple[set[int], set[int]]:
    years = {int(y) for y in _YEAR_RE.findall(question)}
    ql = question.lower()
    months = {n for name, n in _MONTHS.items() if name in ql}
    return years, months


def _load_chunks(path: Path, chunk_chars: int) -> list[str]:
    """Read a corpus txt and split into paragraph-aligned chunks (cached)."""
    key = f"{path}:{chunk_chars}"
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]
    text = path.read_text(encoding="utf-8", errors="ignore")
    paras = text.split("\n\n")
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > chunk_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    _CHUNK_CACHE[key] = chunks
    return chunks


def retrieve_corpus_chunks(question: str, corpus_dir: Path,
                           source_files: list[str], chunk_chars: int,
                           top_chunks: int) -> list[str]:
    """Period-aware retrieval over the question's oracle documents.

    OfficeQA 题目几乎总含明确时间（"as reported in January 1941"）；
    文件名自带年月（treasury_bulletin_1941_01.txt）。检索分两档：
    1. 题目提到的年份能匹配上的文件优先（有匹配则只看匹配文件）
    2. 块打分 = bigram 重合 + 月份命中加权 + 含数字（表格）加权
    """
    years, months = _question_periods(question)
    q = _bigrams(question)
    phrases = _question_phrases(question)

    files: list[tuple[Path, int | None]] = []   # (path, file_month)
    for name in source_files:
        p = Path(corpus_dir) / name
        if not p.is_file():
            continue
        m = _FILE_PERIOD_RE.match(name)
        fy, fm = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        if years and fy is not None and fy not in years:
            continue                              # 年份不符的文件直接排除
        files.append((p, fm))
    if not files:                                 # 年份过滤后为空 → 回退全部
        for name in source_files:
            p = Path(corpus_dir) / name
            if p.is_file():
                m = _FILE_PERIOD_RE.match(name)
                files.append((p, int(m.group(2)) if m else None))

    scored: list[tuple[float, str, str]] = []
    for p, fm in files:
        for chunk in _load_chunks(p, chunk_chars):
            score = float(len(q & _bigrams(chunk)))
            cl = chunk.lower()
            phrase_hits = sum(1 for ph in phrases if ph in cl)
            if phrase_hits:
                score += 10.0 * phrase_hits       # 专名短语精确命中：最强信号
            if score <= 0:
                continue
            if fm is not None and fm in months:
                score *= 1.5                      # 月份命中的文件块加权
            if _DIGIT_RE.search(chunk):
                score *= 1.2                      # 含数字（表格）块加权
            scored.append((score, chunk, p.name))
    # 多文件题目按文件配额分配：避免单文件块占满 top_chunks（跨表题必输）
    by_file: dict[str, list[tuple[float, str]]] = {}
    for score, chunk, fname in scored:
        by_file.setdefault(fname, []).append((score, chunk))
    picked: list[tuple[float, str]] = []
    if len(by_file) > 1:
        per_file = max(2, top_chunks // len(by_file))
        for fname, lst in by_file.items():
            lst.sort(key=lambda x: -x[0])
            picked.extend(lst[:per_file])
        picked.sort(key=lambda x: -x[0])
        return [c for _, c in picked[:top_chunks]]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c, _ in scored[:top_chunks]]


# ---------- agent & judge ----------

_ANSWER_SYSTEM = (
    "你是办公室事务问答助手。若提供了参考技能文档，优先依据其中的事实作答；"
    "否则凭自己的知识作答。只输出简短答案本身，不要解释。"
)

_CORPUS_SYSTEM = (
    "你是金融文档问答助手。依据提供的文档片段与技能文档作答。"
    "技能文档包含解题程序性知识（如何定位表格、处理单位、交叉验证），请遵循。"
    "先简要推理，然后必须以 <FINAL_ANSWER>最终答案</FINAL_ANSWER> 结束。"
    "最终答案保持简洁（数值、货币或列表），不要单位换算 invention。"
)


def answer_question(client: DeepSeekClient, question: str,
                    library_dir: Path, top_k: int = 2,
                    corpus_chunks: list[str] | None = None) -> str:
    skills = retrieve_topk(question, library_dir, k=top_k)
    parts = []
    if corpus_chunks:
        parts.append("文档片段：\n\n" + "\n\n---\n\n".join(corpus_chunks))
    if skills:
        parts.append("参考技能文档：\n\n" + "\n\n---\n\n".join(skills))
    context = ("\n\n".join(parts) + "\n\n") if parts else ""
    system = _CORPUS_SYSTEM if corpus_chunks is not None else _ANSWER_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context}问题：{question}"},
    ]
    return client.chat(messages, max_tokens=1024 if corpus_chunks else 256)


def _norm(text: str) -> str:
    """Judge normalization: case-fold + strip all whitespace.

    模型常输出 '5000 元' 而标准答案为 '5000元'——空白差异不构成错误。
    """
    return re.sub(r"\s+", "", text).lower()


def judge(gold: str, output: str) -> int:
    return int(_norm(gold) in _norm(output))


def make_judge(bench: Benchmark):
    """按 benchmark.scoring 选择判分器：substring（默认）或 OfficeQA 官方模糊判分。"""
    if bench.scoring == "officeqa_fuzzy":
        import importlib.util
        reward_path = bench.root / "vendor" / "reward.py"
        spec = importlib.util.spec_from_file_location("officeqa_reward", reward_path)
        reward = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward)

        def fuzzy(gold: str, output: str) -> int:
            return int(reward.score_answer(gold, output, bench.tolerance) >= 1.0)

        return fuzzy
    return judge


def make_llm_eval_fn(client: DeepSeekClient, bench: Benchmark, top_k: int = 2,
                     max_workers: int = 6):
    """Build a Gate-compatible evaluator: (library_dir, task_ids) -> result dict.

    语料型基准单题上下文大，评测并发执行（结果与顺序无关，确定性不受影响）。
    """
    from concurrent.futures import ThreadPoolExecutor

    task_by_id = {t["id"]: t for t in bench.tasks}
    judge_fn = make_judge(bench)
    corpus_dir = (bench.root / bench.corpus_dir) if bench.corpus_dir else None

    def _score_one(tid: str) -> tuple[str, int]:
        t = task_by_id[tid]
        chunks = None
        if corpus_dir is not None:
            chunks = retrieve_corpus_chunks(
                t["question"], corpus_dir, t.get("source_files", []),
                bench.chunk_chars, bench.top_chunks)
        out = answer_question(client, t["question"], Path(library_dir_ref[0]), top_k,
                              corpus_chunks=chunks)
        return tid, judge_fn(t["answer"], out)

    library_dir_ref: list = [None]  # set per call, read by workers

    def eval_fn(library_dir, task_ids) -> dict:
        library_dir_ref[0] = library_dir
        scores: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for tid, s in pool.map(_score_one, task_ids):
                scores[tid] = s
        n = len(task_ids)
        return {"scores": scores,
                "accuracy": (sum(scores.values()) / n) if n else 0.0, "n": n}

    return eval_fn


# ---------- evolver ----------

_EVOLVER_SYSTEM = (
    "你是 Agent 技能提炼器。根据本轮任务执行轨迹（含模型作答、是否正确、"
    "正确答案）修订技能文档。规则：\n"
    "1. 把被验证为正确的事实性知识条目化保留；答错的题，用正确答案修正或新增条目；\n"
    "2. 删除被轨迹证伪的内容；保持条目简洁、一事一条；\n"
    "3. 只输出修订后的 Markdown 技能正文，以 # 标题开头，不要输出任何解释。"
)


class LLMEvolver:
    """Distills execution trajectories into a revised skill document."""

    name = "llm"

    def __init__(self, client: DeepSeekClient):
        self.client = client

    def revise(self, current_skill_md: str | None,
               trajectories: list[dict]) -> str:
        traj_text = "\n".join(
            f"- 问题：{t['question']}\n  模型作答：{t['output']}\n"
            f"  是否正确：{'是' if t['correct'] else '否'}；正确答案：{t['gold']}"
            for t in trajectories
        )
        user = ""
        if current_skill_md:
            user += f"当前技能文档：\n\n{current_skill_md}\n\n---\n\n"
        user += f"本轮执行轨迹：\n{traj_text}\n\n请输出修订后的技能文档。"
        out = self.client.chat(
            [{"role": "system", "content": _EVOLVER_SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=1500, temperature=0.2,
        ).strip()
        # 兜底：保证通过静态检查的最低格式
        if not out.startswith("#"):
            out = "# 蒸馏技能\n\n" + out
        return out[:8000]


# ---------- experiment runner ----------

@dataclass
class LLMExperimentConfig:
    benchmark: str
    model: str = "deepseek-chat"
    arm: str = "ungoverned"          # "ungoverned" | "gate"
    rounds: int = 3
    train_per_round: int = 4
    seed: int = 1
    min_delta: float = 0.0
    top_k: int = 2
    out: str | None = None
    eval_max_tasks: int = 0          # >0 时截断 eval split（smoke 成本校准用）
    max_workers: int = 6             # API 并发度


def run_llm_experiment(reg: Registry, config: LLMExperimentConfig,
                       client: DeepSeekClient | None = None) -> dict:
    bench = Benchmark.load(config.benchmark)
    if not bench.train_tasks:
        raise ValueError(f"benchmark {bench.name} has no train_file tasks")
    client = client or DeepSeekClient(model=config.model)
    evolver = LLMEvolver(client)
    judge_fn = make_judge(bench)
    eval_fn = make_llm_eval_fn(client, bench, top_k=config.top_k,
                               max_workers=config.max_workers)
    gate = (Gate(bench, min_delta=config.min_delta, eval_fn=eval_fn)
            if config.arm == "gate" else None)
    eval_ids = bench.eval_split()
    if config.eval_max_tasks > 0:
        eval_ids = eval_ids[: config.eval_max_tasks]

    run_id = (f"{utc_now().replace(':', '').replace('+', 'Z')}"
              f"_{config.arm}_llm-{config.model}_s{config.seed}")
    run_dir = Path(config.out) if config.out else reg.root / "runs"
    run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "run_id": run_id,
        "created_at": utc_now(),
        "config": asdict(config),
        "code_version": skillops.__version__,
        "git_head_start": reg.git.head(),
        "benchmark": {"name": bench.name, "eval_tasks": len(eval_ids),
                      "holdout_tasks": len(bench.holdout_split()),
                      "train_tasks": len(bench.train_tasks)},
        "rounds": [],
        "final": {},
        "cost": {},   # filled from client.usage at the end
    }

    current_skill_md: str | None = None
    train = list(bench.train_tasks)
    tmp_dir = Path(tempfile.mkdtemp(prefix="skillops_llmexp_"))
    try:
        for r in range(1, config.rounds + 1):
            batch = train[(r - 1) * config.train_per_round:
                          (r - 1) * config.train_per_round + config.train_per_round]
            if not batch:
                batch = train[: config.train_per_round]

            # 1) Agent 执行训练任务（带当前技能库 + 语料检索，并发），收集轨迹
            from concurrent.futures import ThreadPoolExecutor

            corpus_dir = (bench.root / bench.corpus_dir) if bench.corpus_dir else None
            view = reg.library_view()

            def _run_train(t: dict) -> dict:
                chunks = None
                if corpus_dir is not None:
                    chunks = retrieve_corpus_chunks(
                        t["question"], corpus_dir, t.get("source_files", []),
                        bench.chunk_chars, bench.top_chunks)
                out = answer_question(client, t["question"], view, config.top_k,
                                      corpus_chunks=chunks)
                return {
                    "id": t["id"], "question": t["question"], "output": out,
                    "gold": t["answer"], "correct": judge_fn(t["answer"], out),
                }

            try:
                with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
                    trajectories = list(pool.map(_run_train, batch))
            finally:
                shutil.rmtree(view, ignore_errors=True)
            (run_dir / f"trajectories_r{r}.json").write_text(
                json.dumps(trajectories, ensure_ascii=False, indent=2),
                encoding="utf-8")

            # 2) Evolver 修订技能 → stage
            current_skill_md = evolver.revise(current_skill_md, trajectories)
            skill_file = tmp_dir / f"distilled_skill_r{r}.md"
            skill_file.write_text(current_skill_md, encoding="utf-8")
            v = reg.stage(skill_file, generator=f"llm:{config.model}")

            # 3) 治理决策
            if gate is not None:
                verdict = gate.evaluate(reg, v.vid)
                if verdict.passed:
                    reg.commit(v.vid)
                    decision = "committed"
                else:
                    reg.reject(v.vid, reason=verdict.reason)
                    decision = "rejected"
                    current_skill_md = None   # 被驳回的技能不带入下一轮
            else:
                reg.commit(v.vid)
                decision = "committed"

            # 4) eval split 评一次
            view = reg.library_view()
            try:
                acc = eval_fn(view, eval_ids)["accuracy"]
            finally:
                shutil.rmtree(view, ignore_errors=True)
            manifest["rounds"].append({
                "round": r, "vid": v.vid, "decision": decision,
                "train_batch": [t["id"] for t in batch],
                "train_correct": sum(t["correct"] for t in trajectories),
                "eval_accuracy": acc,
                "library_size": len([x for x in reg.list_versions()
                                     if x.status.value == "committed"]),
            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 锁定 holdout：实验结束时评一次
    view = reg.library_view()
    try:
        hold_acc = eval_fn(view, bench.holdout_split())["accuracy"]
    finally:
        shutil.rmtree(view, ignore_errors=True)

    manifest["final"] = {
        "eval_accuracy": manifest["rounds"][-1]["eval_accuracy"]
        if manifest["rounds"] else 0.0,
        "peak_eval_accuracy": max((x["eval_accuracy"] for x in manifest["rounds"]),
                                  default=0.0),
        "holdout_accuracy": hold_acc,
        "library_size": manifest["rounds"][-1]["library_size"]
        if manifest["rounds"] else 0,
    }
    manifest["git_head_end"] = reg.git.head()
    manifest["cost"] = dict(client.usage)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    reg.ledger.audit("experiment", {
        "run_id": run_id, "arm": config.arm, "evolver": f"llm:{config.model}",
        "seed": config.seed, "rounds": config.rounds,
        "final_eval": manifest["final"]["eval_accuracy"],
        "total_tokens": client.usage["total_tokens"],
    })
    return manifest
