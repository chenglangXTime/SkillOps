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

M2 将提供：`skillops gate run`（自动评测门控）、`skillops sweep`（生命周期巡检）、`skillops scan security`（安全探测）、`skillops rollback`（回滚）。

## 设计文档

完整项目提案（定位、架构、API、里程碑、技术报告大纲）见同工作区的
《SkillOps开源项目提案.md》。

## License

Apache License 2.0
