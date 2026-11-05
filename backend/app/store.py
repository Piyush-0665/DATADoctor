"""Controlled local storage. Only application-produced objects are deserialized."""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def uid():
    return uuid.uuid4().hex


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv('DATADOCTOR_STORAGE', Path(__file__).parents[1] / 'storage')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('datasets', 'models'):
            (self.root / name).mkdir(exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS entities (id TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(kind,id))')
            primary = {r[1] for r in db.execute('PRAGMA table_info(entities)') if r[5]}
            if primary != {'kind','id'}:
                db.execute('ALTER TABLE entities RENAME TO entities_legacy')
                db.execute('CREATE TABLE entities (id TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(kind,id))')
                db.execute('INSERT INTO entities SELECT id,kind,created,payload FROM entities_legacy')
                db.execute('DROP TABLE entities_legacy')

    def connect(self):
        return sqlite3.connect(self.root / 'metadata.sqlite3', timeout=30)

    def put(self, kind, data):
        data = safe_json(data)
        data.setdefault('id', uid())
        data.setdefault('created', datetime.now(timezone.utc).isoformat())
        with self.lock, self.connect() as db:
            db.execute('INSERT INTO entities VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload',
                       (data['id'], kind, data['created'], json.dumps(data, allow_nan=False)))
        return data

    def get(self, kind, identifier):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM entities WHERE id=? AND kind=?', (identifier, kind)).fetchone()
        if not row:
            raise KeyError(f'{kind} not found')
        return json.loads(row[0])

    def all(self, kind):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM entities WHERE kind=? ORDER BY created DESC', (kind,))]

    def delete(self, kind, identifier):
        with self.lock, self.connect() as db:
            db.execute('DELETE FROM entities WHERE kind=? AND id=?', (kind, identifier))

    def path(self, category, identifier, suffix):
        if category not in ('datasets', 'models') or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid artifact identifier')
        return self.root / category / f'{identifier}.{suffix}'

    def frame(self, version_id):
        self.get('version', version_id)
        return pd.read_pickle(self.path('datasets', version_id, 'pkl'))

    def save_frame(self, dataset_id, frame, operations, parent=None, original=False):
        identifier = uid()
        frame.to_pickle(self.path('datasets', identifier, 'pkl'))
        return self.put('version', dict(id=identifier, dataset_id=dataset_id, parent=parent,
                                       operations=operations, original=original, rows=len(frame), columns=len(frame.columns)))

    def recover_jobs(self):
        for job in self.all('job'):
            if job['status'] in ('queued', 'running'):
                job.update(status='failed', error='Server restarted before completion. Submit a new job.', stage='Interrupted')
                self.put('job', job)
