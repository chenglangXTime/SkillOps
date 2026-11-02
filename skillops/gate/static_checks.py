"""Static checks — Gate layer 1: fast, deterministic, fail-fast.

在跑任何基准之前先做静态检查：格式、长度、密钥泄露、占位符比例。
任何 block 级 finding 直接驳回，不消耗基准算力。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_LEN = 100
MAX_LEN = 50_000
MAX_PLACEHOLDER_RATIO = 0.3

_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key material"),
    (re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\"]{12,}"),
     "hardcoded credential assignment"),
]

_PLACEHOLDER = re.compile(r"(?i)\b(TODO|FIXME|XXX|TBD|占位|待补)\b")


@dataclass
class StaticFinding:
    check: str
    severity: str  # "warn" | "block"
    message: str


def run_static_checks(content: str) -> list[StaticFinding]:
    findings: list[StaticFinding] = []

    # 1. 标题：第一个非空行必须是 Markdown 标题
    first_line = next((ln for ln in content.splitlines() if ln.strip()), "")
    if not first_line.lstrip().startswith("#"):
        findings.append(StaticFinding(
            "format.title", "block", "skill must start with a Markdown heading"))

    # 2. 长度
    n = len(content)
    if n < MIN_LEN:
        findings.append(StaticFinding(
            "format.length", "block", f"content too short ({n} < {MIN_LEN} chars)"))
    elif n > MAX_LEN:
        findings.append(StaticFinding(
            "format.length", "block", f"content too long ({n} > {MAX_LEN} chars)"))

    # 3. 密钥/凭据泄露
    for pattern, label in _SECRET_PATTERNS:
        if pattern.search(content):
            findings.append(StaticFinding(
                "security.secret", "block", f"possible leaked secret: {label}"))

    # 4. 占位符比例
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if lines:
        ratio = sum(1 for ln in lines if _PLACEHOLDER.search(ln)) / len(lines)
        if ratio > MAX_PLACEHOLDER_RATIO:
            findings.append(StaticFinding(
                "quality.placeholder", "block",
                f"placeholder ratio {ratio:.0%} > {MAX_PLACEHOLDER_RATIO:.0%}"))
        elif ratio > 0:
            findings.append(StaticFinding(
                "quality.placeholder", "warn",
                f"contains placeholder markers ({ratio:.0%} of lines)"))

    return findings


def has_blocker(findings: list[StaticFinding]) -> bool:
    return any(f.severity == "block" for f in findings)
