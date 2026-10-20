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

# Worker engines; a job's engine follows from its kind (params["kind"], "transcribe" if absent)
TRANSCRIBE, DIARIZE = "transcribe", "diarize"
KIND_ENGINES = {"transcribe": TRANSCRIBE, "tadiff": TRANSCRIBE, "docdiff": TRANSCRIBE, "diarize": DIARIZE}


def kind_of(params: dict) -> str:
    return params.get("kind", "transcribe")


def engine_of(params: dict) -> str:
    return KIND_ENGINES[kind_of(params)]


class JobStore:
    """Job state in SQLite; running jobs belong to a worker until its lease runs out"""

    def __init__(self, db_path: Optional[str] = None, jobs_dir: Optional[str] = None):
        self.jobs_dir = jobs_dir or config.JOBS_DIR
        os.makedirs(self.jobs_dir, exist_ok=True)
        db_path = db_path or config.DB_PATH or os.path.join(self.jobs_dir, "jobs.sqlite")
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
                owner TEXT,
                worker TEXT,
                lease_until REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL,
                started REAL,
                finished REAL,
                stats TEXT
            )""")
        # Older databases lack the lease columns (from before workers were separate) and stats
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(jobs)")}
        for name, ddl in (("owner", "TEXT"), ("worker", "TEXT"), ("lease_until", "REAL"),
                          ("attempts", "INTEGER NOT NULL DEFAULT 0"), ("stats", "TEXT")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")
        self.db.commit()

    def job_dir(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, job_id)

    def create(self, params: dict, source: str, job_id: Optional[str] = None, owner: Optional[str] = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        os.makedirs(self.job_dir(job_id), exist_ok=True)
        with self.lock:
            self.db.execute(
                "INSERT INTO jobs (id, status, params, source, owner, created) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, QUEUED, json.dumps(params), source, owner, time.time()))
            self.db.commit()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["params"] = json.loads(job["params"])
        job["stats"] = json.loads(job["stats"]) if job["stats"] else None
        return job

    def update(self, job_id: str, **fields):
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            self.db.commit()

    def list(self, owner: Optional[str] = None, limit: int = 50) -> list:
        """Newest first; owner None = everyone's"""
        where, args = ("WHERE owner = ?", (owner,)) if owner is not None else ("", ())
        with self.lock:
            rows = self.db.execute(f"SELECT id FROM jobs {where} ORDER BY created DESC LIMIT ?",
                                   (*args, limit)).fetchall()
        return [self.get(r["id"]) for r in rows]

    def queued_ids(self) -> list:
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created", (QUEUED,)).fetchall()
        return [r["id"] for r in rows]

    def queue_position(self, job_id: str) -> Optional[int]:
        ids = self.queued_ids()
        return ids.index(job_id) + 1 if job_id in ids else None

    def lease(self, worker: str, langs: list, lease_s: float, engines=(TRANSCRIBE,)) -> Optional[dict]:
        """Give the oldest queued job this worker can do to it: a transcription in one of its
        languages, or a job for another of its engines"""
        now = time.time()
        with self.lock:
            rows = self.db.execute(
                "SELECT id, params FROM jobs WHERE status = ? ORDER BY created", (QUEUED,)).fetchall()
            for row in rows:
                params = json.loads(row["params"])
                engine = engine_of(params)
                if engine in engines and (engine != TRANSCRIBE or params["lang"] in langs):
                    self.db.execute(
                        "UPDATE jobs SET status = ?, worker = ?, lease_until = ?, attempts = attempts + 1, "
                        "stage = NULL, progress = 0, started = ? WHERE id = ?",
                        (RUNNING, worker, now + lease_s, now, row["id"]))
                    self.db.commit()
                    break
            else:
                return None
        return self.get(row["id"])

    def owned(self, job_id: str, worker: str) -> Optional[dict]:
        """The job if it is still running on this worker (not cancelled, not re-leased)"""
        job = self.get(job_id)
        if job and job["status"] == RUNNING and job["worker"] == worker:
            return job
        return None

    def heartbeat(self, job_id: str, worker: str, stage: Optional[str], progress: float, lease_s: float) -> bool:
        """Extend the lease; False if the job is no longer this worker's (cancelled, expired)"""
        with self.lock:
            cur = self.db.execute(
                "UPDATE jobs SET stage = ?, progress = ?, lease_until = ? "
                "WHERE id = ? AND status = ? AND worker = ?",
                (stage, progress, time.time() + lease_s, job_id, RUNNING, worker))
            self.db.commit()
        return cur.rowcount == 1

    def expire_leases(self, max_attempts: int) -> list:
        """Running jobs whose worker went silent: back to the queue, or error after max_attempts"""
        now = time.time()
        with self.lock:
            rows = self.db.execute(
                "SELECT id, attempts FROM jobs WHERE status = ? AND lease_until < ?", (RUNNING, now)).fetchall()
            for row in rows:
                if row["attempts"] >= max_attempts:
                    self.db.execute(
                        "UPDATE jobs SET status = ?, error = ?, finished = ? WHERE id = ?",
                        (ERROR, f"worker lost {row['attempts']} times", now, row["id"]))
                else:
                    self.db.execute(
                        "UPDATE jobs SET status = ?, worker = NULL, lease_until = NULL, stage = NULL, "
                        "progress = 0 WHERE id = ?", (QUEUED, row["id"]))
            self.db.commit()
        return [r["id"] for r in rows]

    def cleanup(self, ttl_hours: Optional[float] = None) -> int:
        """Delete finished jobs older than TTL together with their files"""
        cutoff = time.time() - (config.JOB_TTL_HOURS if ttl_hours is None else ttl_hours) * 3600
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
