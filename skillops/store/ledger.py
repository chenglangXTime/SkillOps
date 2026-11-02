"""SQLite ledger: version records + audit log. (DuckDB MetricStore arrives in M2.)"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from skillops.core.models import SkillStatus, SkillVersion, utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_versions (
    vid TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_ids TEXT NOT NULL DEFAULT '[]',
    source_trajectories TEXT NOT NULL DEFAULT '[]',
    generator TEXT NOT NULL DEFAULT '',
    metrics TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_id ON skill_versions(skill_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}'
);
"""


class Ledger:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---- skill_versions ----

    def insert_version(self, v: SkillVersion) -> None:
        self.conn.execute(
            "INSERT INTO skill_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                v.vid, v.skill_id, v.version, v.name, v.rel_path, v.sha256,
                v.status.value, json.dumps(v.parent_ids, ensure_ascii=False),
                json.dumps(v.source_trajectories, ensure_ascii=False),
                v.generator, json.dumps(v.metrics, ensure_ascii=False), v.created_at,
            ),
        )
        self.conn.commit()

    def update_status(self, vid: str, status: SkillStatus, metrics: dict | None = None) -> None:
        if metrics is not None:
            self.conn.execute(
                "UPDATE skill_versions SET status=?, metrics=? WHERE vid=?",
                (status.value, json.dumps(metrics, ensure_ascii=False), vid),
            )
        else:
            self.conn.execute(
                "UPDATE skill_versions SET status=? WHERE vid=?", (status.value, vid)
            )
        self.conn.commit()

    def next_version(self, skill_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS mv FROM skill_versions WHERE skill_id=?",
            (skill_id,),
        ).fetchone()
        return int(row["mv"]) + 1

    def latest_parents(self, skill_id: str) -> list[str]:
        row = self.conn.execute(
            "SELECT vid FROM skill_versions WHERE skill_id=? AND status=? "
            "ORDER BY version DESC LIMIT 1",
            (skill_id, SkillStatus.COMMITTED.value),
        ).fetchone()
        return [row["vid"]] if row else []

    def get(self, vid: str) -> SkillVersion | None:
        row = self.conn.execute(
            "SELECT * FROM skill_versions WHERE vid=?", (vid,)
        ).fetchone()
        return self._row_to_version(row) if row else None

    def list_versions(self, status: SkillStatus | None = None) -> list[SkillVersion]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM skill_versions ORDER BY created_at, skill_id, version"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM skill_versions WHERE status=? ORDER BY created_at, skill_id, version",
                (status.value,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> SkillVersion:
        return SkillVersion(
            skill_id=row["skill_id"],
            version=row["version"],
            name=row["name"],
            rel_path=row["rel_path"],
            sha256=row["sha256"],
            status=SkillStatus(row["status"]),
            parent_ids=json.loads(row["parent_ids"]),
            source_trajectories=json.loads(row["source_trajectories"]),
            generator=row["generator"],
            metrics=json.loads(row["metrics"]),
            created_at=row["created_at"],
        )

    # ---- audit_log ----

    def audit(self, action: str, detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (ts, action, detail) VALUES (?,?,?)",
            (utc_now(), action, json.dumps(detail or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def events(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, action, detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"ts": r["ts"], "action": r["action"], "detail": json.loads(r["detail"])}
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()
