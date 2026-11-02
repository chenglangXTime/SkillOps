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


# ---------- agent & judge ----------

_ANSWER_SYSTEM = (
    "你是办公室事务问答助手。若提供了参考技能文档，优先依据其中的事实作答；"
    "否则凭自己的知识作答。只输出简短答案本身，不要解释。"
)


def answer_question(client: DeepSeekClient, question: str,
                    library_dir: Path, top_k: int = 2) -> str:
    skills = retrieve_topk(question, library_dir, k=top_k)
    context = ""
    if skills:
        context = "参考技能文档：\n\n" + "\n\n---\n\n".join(skills) + "\n\n"
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": f"{context}问题：{question}"},
    ]
    return client.chat(messages, max_tokens=256)


def _norm(text: str) -> str:
    """Judge normalization: case-fold + strip all whitespace.

    模型常输出 '5000 元' 而标准答案为 '5000元'——空白差异不构成错误。
    """
    return re.sub(r"\s+", "", text).lower()


def judge(gold: str, output: str) -> int:
    return int(_norm(gold) in _norm(output))


def make_llm_eval_fn(client: DeepSeekClient, bench: Benchmark, top_k: int = 2):
    """Build a Gate-compatible evaluator: (library_dir, task_ids) -> result dict."""
    task_by_id = {t["id"]: t for t in bench.tasks}

    def eval_fn(library_dir, task_ids) -> dict:
        scores = {}
        for tid in task_ids:
            t = task_by_id[tid]
            out = answer_question(client, t["question"], Path(library_dir), top_k)
            scores[tid] = judge(t["answer"], out)
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


def run_llm_experiment(reg: Registry, config: LLMExperimentConfig,
                       client: DeepSeekClient | None = None) -> dict:
    bench = Benchmark.load(config.benchmark)
    if not bench.train_tasks:
        raise ValueError(f"benchmark {bench.name} has no train_file tasks")
    client = client or DeepSeekClient(model=config.model)
    evolver = LLMEvolver(client)
    eval_fn = make_llm_eval_fn(client, bench, top_k=config.top_k)
    gate = (Gate(bench, min_delta=config.min_delta, eval_fn=eval_fn)
            if config.arm == "gate" else None)
    eval_ids = bench.eval_split()

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

            # 1) Agent 执行训练任务（带当前技能库），收集轨迹
            trajectories = []
            for t in batch:
                view = reg.library_view()
                try:
                    out = answer_question(client, t["question"], view, config.top_k)
                finally:
                    shutil.rmtree(view, ignore_errors=True)
                trajectories.append({
                    "id": t["id"], "question": t["question"], "output": out,
                    "gold": t["answer"], "correct": judge(t["answer"], out),
                })
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
