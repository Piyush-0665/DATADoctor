import sqlite3

from app.store import Store


def test_record_types_can_share_a_job_id(tmp_path):
    store=Store(tmp_path)
    job=store.put('job',dict(status='running'))
    store.put('profile',dict(id=job['id'],missing_total=4))
    store.put('experiment',dict(id=job['id'],status='completed'))
    assert store.get('job',job['id'])['status']=='running'
    assert store.get('profile',job['id'])['missing_total']==4
    assert store.get('experiment',job['id'])['status']=='completed'


def test_legacy_schema_migration_preserves_records(tmp_path):
    with sqlite3.connect(tmp_path/'metadata.sqlite3') as db:
        db.execute('CREATE TABLE entities (id TEXT PRIMARY KEY,kind TEXT NOT NULL,created TEXT NOT NULL,payload TEXT NOT NULL)')
        db.execute('INSERT INTO entities VALUES (?,?,?,?)',('test','dataset','2026-01-01','{"id":"test","name":"retained.csv"}'))
    store=Store(tmp_path)
    assert store.get('dataset','test')['name']=='retained.csv'
    store.put('job',dict(id='test',status='queued'))
    assert store.get('dataset','test')['name']=='retained.csv'
