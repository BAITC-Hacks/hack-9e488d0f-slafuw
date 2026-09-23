"""Persistent bounded result/session store and sanitized tool audit, standard-library SQLite."""
import json
from pathlib import Path
import secrets
import sqlite3
from threading import RLock
import time


class Store:
    def __init__(self, path):
        if path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS objects (
                kind TEXT, id TEXT, body TEXT, updated REAL, PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY, body TEXT, created REAL);
        ''')
        self.db.commit()

    def put(self, kind, key, body):
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?,?)',
                            (kind, key, json.dumps(body, ensure_ascii=False), time.time()))
            self.db.execute('DELETE FROM objects WHERE updated < ?', (time.time() - 86400,))
            self.db.execute('DELETE FROM objects WHERE rowid IN (SELECT rowid FROM objects '
                            'ORDER BY updated DESC LIMIT -1 OFFSET 2000)')

    def get(self, kind, key):
        if not isinstance(key, str):
            raise ValueError('Требуется строковый идентификатор')
        with self.lock:
            row = self.db.execute('SELECT body FROM objects WHERE kind=? AND id=? AND updated>=?',
                                  (kind, key, time.time() - 86400)).fetchone()
        if not row:
            raise ValueError('Результат или сессия не найдены либо истекли')
        return json.loads(row[0])

    def session(self, session_id):
        if session_id:
            return session_id, self.get('session', session_id)
        key = secrets.token_urlsafe(24)
        value = {'request': {}, 'last_result_id': None, 'revision': 0}
        self.put('session', key, value)
        return key, value

    def save_session(self, key, value, revision):
        with self.lock:
            previous = self.get('session', key)
            if previous['revision'] != revision:
                raise ValueError('Сессия изменена другим запросом; повторите сообщение')
            value['revision'] = revision + 1
            self.put('session', key, value)

    def audit(self, value):
        with self.lock, self.db:
            self.db.execute('INSERT INTO audit(body,created) VALUES (?,?)',
                            (json.dumps(value, ensure_ascii=False), time.time()))
            self.db.execute('DELETE FROM audit WHERE id NOT IN '
                            '(SELECT id FROM audit ORDER BY id DESC LIMIT 5000)')

    def close(self):
        self.db.close()
