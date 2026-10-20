import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Optional

from . import config

QUEUED, RUNNING, DONE, ERROR, CANCELLED = "queued", "running", "done", "error", "cancelled"
FINAL = (DONE, ERROR, CANCELLED)


class JobStore:
    """Job state in SQLite, so queued jobs survive a restart"""

    def __init__(self, db_path: str = config.DB_PATH, jobs_dir: str = config.JOBS_DIR):
        self.jobs_dir = jobs_dir
        os.makedirs(jobs_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT,
                progress REAL NOT NULL DEFAULT 0,
                params TEXT NOT NULL,
                source TEXT NOT NULL,
                error TEXT,
                created REAL NOT NULL,
                started REAL,
                finished REAL
            )""")
        self.db.commit()

    def job_dir(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, job_id)

    def create(self, params: dict, source: str, job_id: Optional[str] = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        os.makedirs(self.job_dir(job_id), exist_ok=True)
        with self.lock:
            self.db.execute(
                "INSERT INTO jobs (id, status, params, source, created) VALUES (?, ?, ?, ?, ?)",
                (job_id, QUEUED, json.dumps(params), source, time.time()))
            self.db.commit()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["params"] = json.loads(job["params"])
        return job

    def update(self, job_id: str, **fields):
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            self.db.commit()

    def queued_ids(self) -> list:
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created", (QUEUED,)).fetchall()
        return [r["id"] for r in rows]

    def queue_position(self, job_id: str) -> Optional[int]:
        ids = self.queued_ids()
        return ids.index(job_id) + 1 if job_id in ids else None

    def recover(self):
        """Jobs that were running when the process died cannot be resumed"""
        with self.lock:
            self.db.execute(
                "UPDATE jobs SET status = ?, error = ?, finished = ? WHERE status = ?",
                (ERROR, "interrupted by server restart", time.time(), RUNNING))
            self.db.commit()

    def cleanup(self, ttl_hours: float = config.JOB_TTL_HOURS) -> int:
        """Delete finished jobs older than TTL together with their files"""
        cutoff = time.time() - ttl_hours * 3600
        with self.lock:
            rows = self.db.execute(
                f"SELECT id FROM jobs WHERE status IN ({','.join('?' * len(FINAL))}) AND finished < ?",
                (*FINAL, cutoff)).fetchall()
            ids = [r["id"] for r in rows]
            self.db.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
            self.db.commit()
        for job_id in ids:
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return len(ids)
