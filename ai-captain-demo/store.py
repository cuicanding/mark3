"""ConfigStore + SQLite 数据层。

设计要点：
- SQLite，stdlib sqlite3，并发由 FastAPI 的线程池/actor 模型保证。
- 所有可变配置写库，支持版本号热生效：改配置后 revision 递增，不重启。
- 生物特征向量（人脸/声纹）仍存 JSON 文件，库里只存成员元数据；实现"只存向量"红线。
- 提供 typed getter/setter，settings 兜底默认值，避免 key 不存在崩溃。
"""

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "captain.db"

DEFAULT_SETTINGS: dict[str, str] = {
    "face_threshold": "0.45",
    "voiceprint_threshold": "0.55",
    "face_min_size": "120",
    "audio_mode": "ptt",           # ptt | voiceprint | hybrid
    "voiceprint_enabled": "false",
    "tts_engine": "melo",          # melo | kokoro
    "tts_speed": "1.0",
    "tts_kokoro_voice": "af",      # kokoro 默认音色
    "tts_fall_back_on_delay": "true",
    "default_persona": "default",
    "interrupt_mode": "false",
}


@dataclass
class Member:
    id: int
    name: str
    permission_flags: int = 0
    created_at: float = 0.0


@dataclass
class RosterRow:
    id: int
    team: str
    task: str
    owner: str
    backup: str
    progress: int
    status: str
    created_at: float
    updated_at: float


@dataclass
class PlaybookRow:
    id: int
    text: str
    cron: str
    audio_path: str
    enabled: int
    created_at: float
    updated_at: float


class ConfigStore:
    """单例配置存储。所有 public 方法线程安全。"""

    _instance_lock = threading.Lock()
    _instance: "ConfigStore | None" = None

    def __new__(cls) -> "ConfigStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if getattr(self, "_init", False):
            return
        self._init = True
        self._lock = threading.RLock()
        self._revision = 0
        self._ensure_schema()
        self._init_defaults()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    permission_flags INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS roster (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    task TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    backup TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    status TEXT DEFAULT '进行中',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS playbook (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    cron TEXT NOT NULL,
                    audio_path TEXT DEFAULT '',
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS personas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    system_prompt TEXT NOT NULL,
                    is_default INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS model_provider (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    base_url TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    timeout_ms INTEGER DEFAULT 1500,
                    is_default INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    enabled INTEGER DEFAULT 1,
                    config TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    template TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    def _init_defaults(self):
        with self._lock:
            with self._conn() as conn:
                for k, v in DEFAULT_SETTINGS.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO settings(key,value,updated_at) VALUES(?,?,?)",
                        (k, v, time.time()),
                    )
                # 默认人格
                conn.execute(
                    "INSERT OR IGNORE INTO personas(name,system_prompt,is_default) VALUES(?,?,1)",
                    (
                        "default",
                        "你是大促作战室的 AI 队长。回答必须简短、口语化，不超过 50 字。"
                        "只能依据工具返回的数据作答，没有数据就说不知道。",
                    ),
                )
                # 默认 LLM provider 占位（不会真正调用）
                conn.execute(
                    "INSERT OR IGNORE INTO model_provider(name,base_url,api_key,model,is_default) VALUES(?,?,?,?,1)",
                    ("deepseek", "https://api.deepseek.com/v1", "", "deepseek-chat"),
                )
            self._bump_rev()

    def _bump_rev(self):
        with self._lock:
            self._revision += 1

    def revision(self) -> int:
        """调用方可轮询 revision 实现配置热生效。"""
        with self._lock:
            return self._revision

    # ---------------- settings ----------------
    def get(self, key: str, default: str | None = None) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else (default or "")

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key))
        except Exception:
            return default

    def get_bool(self, key: str) -> bool:
        return self.get(key).lower() in ("1", "true", "yes", "on")

    def set(self, key: str, value: Any):
        s = str(value)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, s, time.time()),
            )
        self._bump_rev()

    def settings_dict(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key,value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ---------------- members ----------------
    def member_add(self, name: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO members(name,created_at) VALUES(?,?) ON CONFLICT(name) DO NOTHING",
                (name, time.time()),
            )
            if cur.rowcount == 0:
                row = conn.execute("SELECT id FROM members WHERE name=?", (name,)).fetchone()
                return row["id"]
        self._bump_rev()
        return cur.lastrowid

    def member_list(self) -> list[Member]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM members ORDER BY created_at").fetchall()
        return [Member(**dict(r)) for r in rows]

    def member_delete(self, member_id: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM members WHERE id=?", (member_id,))
        self._bump_rev()
        return True

    # ---------------- roster ----------------
    def roster_list(self) -> list[RosterRow]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM roster ORDER BY id").fetchall()
        return [RosterRow(**dict(r)) for r in rows]

    def roster_add(self, team: str, task: str, owner: str, backup: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO roster(team,task,owner,backup,progress,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (team, task, owner, backup, 0, "进行中", time.time(), time.time()),
            )
        self._bump_rev()
        return cur.lastrowid

    def roster_update(self, row_id: int, **fields) -> bool:
        allowed = {"team", "task", "owner", "backup", "progress", "status"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE roster SET " + ", ".join(f"{k}=?" for k in updates) + " WHERE id=?",
                (*updates.values(), row_id),
            )
        self._bump_rev()
        return True

    def roster_delete(self, row_id: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM roster WHERE id=?", (row_id,))
        self._bump_rev()
        return True

    # ---------------- playbook ----------------
    def playbook_list(self) -> list[PlaybookRow]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM playbook ORDER BY id").fetchall()
        return [PlaybookRow(**dict(r)) for r in rows]

    def playbook_add(self, text: str, cron: str, audio_path: str = "", enabled: int = 1) -> int:
        ts = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO playbook(text,cron,audio_path,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (text, cron, audio_path, enabled, ts, ts),
            )
        self._bump_rev()
        return cur.lastrowid

    def playbook_update(self, pid: int, **fields) -> bool:
        allowed = {"text", "cron", "audio_path", "enabled"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE playbook SET " + ", ".join(f"{k}=?" for k in updates) + " WHERE id=?",
                (*updates.values(), pid),
            )
        self._bump_rev()
        return True

    def playbook_delete(self, pid: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM playbook WHERE id=?", (pid,))
        self._bump_rev()
        return True

    # ---------------- personas ----------------
    def persona_get_default(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM personas WHERE is_default=1 LIMIT 1").fetchone()
            if not row:
                row = conn.execute("SELECT * FROM personas LIMIT 1").fetchone()
        return dict(row) if row else {"name": "default", "system_prompt": ""}

    def persona_list(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM personas ORDER BY id").fetchall()]

    def persona_upsert(self, name: str, system_prompt: str, is_default: bool = False) -> int:
        with self._conn() as conn:
            if is_default:
                conn.execute("UPDATE personas SET is_default=0")
            conn.execute(
                "INSERT INTO personas(name,system_prompt,is_default) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET system_prompt=excluded.system_prompt,"
                "is_default=excluded.is_default",
                (name, system_prompt, int(is_default)),
            )
            row = conn.execute("SELECT id FROM personas WHERE name=?", (name,)).fetchone()
        self._bump_rev()
        return row["id"]

    # ---------------- model_provider ----------------
    def provider_default(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM model_provider WHERE is_default=1 LIMIT 1").fetchone()
            if not row:
                row = conn.execute("SELECT * FROM model_provider LIMIT 1").fetchone()
        return dict(row) if row else {}

    def provider_list(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM model_provider ORDER BY id").fetchall()]

    def provider_upsert(self, name: str, base_url: str, api_key: str, model: str, timeout_ms: int = 1500, is_default: bool = False) -> int:
        with self._conn() as conn:
            if is_default:
                conn.execute("UPDATE model_provider SET is_default=0")
            conn.execute(
                "INSERT INTO model_provider(name,base_url,api_key,model,timeout_ms,is_default) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                "base_url=excluded.base_url, api_key=excluded.api_key, model=excluded.model, "
                "timeout_ms=excluded.timeout_ms, is_default=excluded.is_default",
                (name, base_url, api_key, model, timeout_ms, int(is_default)),
            )
            row = conn.execute("SELECT id FROM model_provider WHERE name=?", (name,)).fetchone()
        self._bump_rev()
        return row["id"]

    # ---------------- intents ----------------
    def intent_list(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM intents ORDER BY priority DESC, updated_at DESC").fetchall()]

    def intent_add(self, name: str, pattern: str, template: str, priority: int = 0) -> int:
        ts = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO intents(name,pattern,template,priority,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (name, pattern, template, priority, 1, ts, ts),
            )
        self._bump_rev()
        return cur.lastrowid

    def intent_update(self, iid: int, **fields) -> bool:
        allowed = {"name", "pattern", "template", "priority", "enabled"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE intents SET " + ", ".join(f"{k}=?" for k in updates) + " WHERE id=?",
                (*updates.values(), iid),
            )
        self._bump_rev()
        return True

    def intent_delete(self, iid: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM intents WHERE id=?", (iid,))
        self._bump_rev()
        return True


    # ---------------- skills ----------------
    def skill_list(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM skills ORDER BY id").fetchall()]

    def skill_set(self, name: str, enabled: int | None = None, config: dict | None = None) -> int:
        with self._conn() as conn:
            ex = conn.execute("SELECT id FROM skills WHERE name=?", (name,)).fetchone()
            if ex:
                sets = []
                vals = []
                if enabled is not None:
                    sets.append("enabled=?")
                    vals.append(enabled)
                if config is not None:
                    sets.append("config=?")
                    vals.append(json.dumps(config, ensure_ascii=False))
                if sets:
                    vals.append(ex["id"])
                    conn.execute("UPDATE skills SET " + ", ".join(sets) + " WHERE id=?", vals)
                sid = ex["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO skills(name,enabled,config) VALUES(?,?,?)",
                    (name, 1 if enabled is None else enabled, json.dumps(config or {}, ensure_ascii=False)),
                )
                sid = cur.lastrowid
        self._bump_rev()
        return sid


store = ConfigStore()
