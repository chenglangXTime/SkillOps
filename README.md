# SkillOps

**Lifecycle governance for self-evolving agent skill libraries.**
自进化 Agent 技能库的生命周期治理框架 —— 技能进化领域的 "MLflow + 安全门"。

SkillOps 不造新的技能进化算法。它为所有技能进化系统（EvoSkill / AutoSkill / MemSkill / Hermes / Claude Code Skills……）提供缺失的治理层：

- **版本化与谱系**：每条技能记录「哪条轨迹 → 哪个版本 → 哪个任务指标」，可 diff、可审计
- **评测门控**：新技能必须通过可复现的对照评测才能入库，否则只能停留在 staging 区
- **退化防控**：漂移检测、自动退役、合并去重（M2 里程碑）
- **安全监测**：沿 Misevolution 四条路径（行为漂移 / 记忆虚假关联 / 工具漏洞 / 工作流放大）的内置探测器（M2）
- **可回滚**：任何一次技能库变更都可以一键撤销

## 安装

```bash
pip install -e .
```

零第三方运行时依赖（仅 Python 3.10+ 标准库 + git）。

## 快速开始

```bash
skillops init my_skill_repo          # 初始化技能库（git + 账本）
cd my_skill_repo

skillops stage path/to/my_skill.md \
  --from traj_001 traj_002 \
  --generator evoskill-v0.3          # 登记候选技能（staging）

skillops status                      # 查看技能库状态
skillops commit <skill_id>           # 通过门控后入库（M1：手动门控）
skillops log                         # 查看版本谱系与审计记录
```

## 评测门控（Gate，M2 已交付第一层）

新技能入库前必须通过三层门控：

```bash
skillops gate office_policy@v1 --bench benchmarks/officeqa_mini --min-delta 0.0
```

1. **静态检查**：格式、长度、密钥泄露、占位符比例 —— 零成本 fail-fast
2. **小样本快筛**：quick_sample 上对比 baseline / candidate，明显回退提前驳回
3. **完整基准 + 锁定 holdout 认证**：全量评估 split 上算 delta，再在锁定 holdout
   上做一次认证，确认增益可迁移、非过拟合

所有门控指标写入账本，`skillops log` 可审计；未通过门控的技能只能留在 staging。

仓库内置 `benchmarks/officeqa_mini/` 演示基准：一个确定性的检索式验证域
（零 LLM 调用、可复现），用于演示门控机制。接入真实 Agent 运行时只需实现
同样的 harness 协议（输入技能库目录 + 任务子集，输出 JSON 分数）。

输出示例：

```
[static]   PASS  (0 findings)
[quick   ] baseline 0.500 → candidate 1.000  (delta +0.500, 4 tasks)
[full    ] baseline 0.667 → candidate 1.000  (delta +0.333, 9 tasks)
[holdout ] baseline 0.000 → candidate 1.000  (delta +1.000, 3 tasks)

verdict: PASS — full delta +0.333, holdout delta +1.000 (required >= +0.000)
```

M2 后续将提供：`skillops sweep`（生命周期巡检）、`skillops scan security`
（安全探测）、`skillops rollback`（回滚）。

## 实验装置（RQ1/RQ2 验证）

内置进化实验环，用于验证"无治理技能库必然退化"（RQ1）及治理臂效果（RQ2 消融）：

```bash
skillops experiment run --bench benchmarks/officeqa_mini \
  --arm ungoverned --rounds 10 --seed 7     # 对照组：所有候选直接入库
skillops experiment run --bench benchmarks/officeqa_mini \
  --arm gate --rounds 10 --seed 7           # 治理臂：候选须过三层门控
```

同一确定性 evolver（`ScriptedEvolver`：先产有用技能、后产高关键词重合的
过期"经验"，通过检索干扰拉低准确率）下的实测对照：

| 臂 | 峰值 → 最终 (eval) | 锁定 holdout | 库大小 |
|---|---|---|---|
| ungoverned | 1.000 → 0.556 | 0.000 | 10（膨胀） |
| gate | 1.000 → 1.000 | 1.000 | 3（精炼） |

每次运行产生不可变 **RunManifest**（`runs/<run_id>/manifest.json`）：配置快照、
随机种子、代码版本、git HEAD、逐轮决策与准确率、最终指标（holdout 全程只在
结束时评一次）、成本台账（LLM token 数，scripted 为 0）。论文真实实验将使用
`LLMEvolver`（M3，对接任意 OpenAI 兼容 API）在真实任务基准上运行同一装置。

> 注：`ScriptedEvolver` + `officeqa_mini` 是**测量装置的可行性验证**，证明退化
> 通道（检索干扰）可被本框架检测与拦截；它不构成论文证据本身。

## 真实 LLM 进化环（`--evolver llm`）

三个组件全部换成真实模型：Agent（DeepSeek 答题，上下文 = 检索到的 top-k 技能）、
Evolver（DeepSeek 读取含对错反馈的执行轨迹、修订技能文档）、评测（同一 Agent
在 eval split 上作答，判分 = 空白不敏感的答案字符串匹配）。

```bash
# 密钥配置（一次性）：把 DEEPSEEK_API_KEY=sk-... 追加到 ~/.env
skillops experiment run --bench benchmarks/officeqa_mini \
  --evolver llm --model deepseek-chat --arm ungoverned --rounds 3 --seed 1
```

Smoke run 实测（deepseek-chat，3 轮 × 4 训练题，45 次 API 调用）：

- 学习曲线：eval 0.667 → 0.889 → 1.000，锁定 holdout 1.000
  （训练题是评测题的改写版，技能经 evolver 蒸馏后跨表述迁移）
- **成本：12,227 tokens ≈ ¥0.02 / 次完整运行**——多 seed × 多臂的完整实验
  预算在个位数人民币量级
- RunManifest 记录模型名、逐轮轨迹（`trajectories_r*.json` 全量留存）、
  token 台账；密钥只从 `~/.env` 读取，永不入仓库或 manifest

判分说明：`judge()` 做大小写折叠 + 去除全部空白后的包含匹配
（模型习惯输出 "5000 元"，标准答案为 "5000元"，空白差异不算错误）。

## 设计文档

完整项目提案（定位、架构、API、里程碑、技术报告大纲）见同工作区的
《SkillOps开源项目提案.md》。

## License

Apache License 2.0
