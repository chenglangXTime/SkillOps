"""Security probes implementing the four Misevolution failure paths. (M2)

| 探测器                   | 检查内容                                   |
|-------------------------|--------------------------------------------|
| behavior_drift          | 入库前后同一 Agent 在拒答基准上的行为对拍  |
| memory_spurious         | 记忆/经验条目中的虚假关联检测              |
| tool_audit              | 自建工具静态扫描（危险调用、外联、凭据）   |
| workflow_amplification  | 工作流变更后单点失败影响面分析             |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Finding:
    probe: str
    severity: str  # "info" | "warn" | "block"
    subject: str   # vid / file path / workflow node
    message: str


class SecurityProbe(Protocol):
    name: str

    def scan(self, registry) -> list[Finding]: ...


class BehaviorDriftProbe:  # pragma: no cover - M2
    name = "behavior_drift"

    def scan(self, registry) -> list[Finding]:
        raise NotImplementedError("Security probes ship in M2.")


class MemorySpuriousProbe:  # pragma: no cover - M2
    name = "memory_spurious"

    def scan(self, registry) -> list[Finding]:
        raise NotImplementedError("Security probes ship in M2.")


class ToolAuditProbe:  # pragma: no cover - M2
    name = "tool_audit"

    def scan(self, registry) -> list[Finding]:
        raise NotImplementedError("Security probes ship in M2.")


class WorkflowAmplificationProbe:  # pragma: no cover - M2
    name = "workflow_amplification"

    def scan(self, registry) -> list[Finding]:
        raise NotImplementedError("Security probes ship in M2.")


DEFAULT_PROBES = [
    BehaviorDriftProbe,
    MemorySpuriousProbe,
    ToolAuditProbe,
    WorkflowAmplificationProbe,
]
