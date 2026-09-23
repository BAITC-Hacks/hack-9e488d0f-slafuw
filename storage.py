"""Durable structured orders, immutable recommendation snapshots and action journal."""

import json
import sqlite3
from threading import RLock
from uuid import uuid4


class Store:
    def __init__(self, path=":memory:"):
        self.lock = RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS actions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL, body TEXT NOT NULL);
        """)

    def put(self, table, key, value):
        if table not in ("results", "sessions"):
            raise ValueError("Unknown store")
        with self.lock, self.db:
            self.db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
                            (key, json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def get(self, table, key):
        if table not in ("results", "sessions"):
            raise ValueError("Unknown store")
        with self.lock:
            row = self.db.execute(f"SELECT body FROM {table} WHERE id = ?", (key,)).fetchone()
        if row is None:
            raise ValueError("Сохранённый результат или диалог не найден")
        return json.loads(row[0])

    def session(self, session_id=None):
        if session_id is not None:
            return session_id, self.get("sessions", session_id)
        session_id = uuid4().hex
        state = {"order": {}, "result_ids": [], "last_response": None}
        self.put("sessions", session_id, state)
        return session_id, state

    def log(self, trace_id, value):
        with self.lock, self.db:
            self.db.execute("INSERT INTO actions (trace_id, body) VALUES (?, ?)",
                            (trace_id, json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def trace(self, trace_id):
        with self.lock:
            rows = self.db.execute("SELECT body FROM actions WHERE trace_id = ? ORDER BY sequence",
                                   (trace_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self):
        with self.lock:
            self.db.close()
