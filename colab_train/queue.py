"""Shared job queue for Colab runs: several Claude sessions can submit at once.

The queue is one JSON file (~/.cache/colab-train/queue.json, override COLAB_TRAIN_QUEUE). Every read-modify-write runs
under an exclusive flock on queue.lock and writes a temp file that is renamed into place, so two sessions never lose an
item. Network calls (W&B, the browser) never happen while the lock is held.

Records hold only: job reference, run name, priority, status, progress numbers and the notify pane. Never a secret.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

ACTIVE = ("queued", "running", "stalled")
TERMINAL = ("finished", "failed", "cancelled", "needs-human")


def queue_path() -> Path:
    override = os.environ.get("COLAB_TRAIN_QUEUE")
    return Path(override) if override else Path.home() / ".cache" / "colab-train" / "queue.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _empty() -> dict:
    return {"items": [], "blocked_until": None}


@contextlib.contextmanager
def locked(path: Path | None = None) -> Iterator[None]:
    p = path or queue_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_name(p.name + ".lock")
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return _empty()
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return _empty()
    data.setdefault("blocked_until", None)
    return data


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def update(fn: Callable[[dict], Any], path: Path | None = None) -> Any:
    """Run fn(data) under the lock; fn may change data in place. Returns fn's result."""
    p = path or queue_path()
    with locked(p):
        data = _read(p)
        result = fn(data)
        _write(p, data)
        return result


def snapshot(path: Path | None = None) -> dict:
    """A copy of the queue for reading. Takes the lock so a half-written file is never seen."""
    p = path or queue_path()
    with locked(p):
        return _read(p)


def find(data: dict, item_id: str) -> dict | None:
    return next((it for it in data["items"] if it["id"] == item_id), None)


def add(job: str, run_name: str | None, priority: int = 0, notify_pane: str | None = None,
        workdir: str | None = None, path: Path | None = None, name: str | None = None) -> dict:
    """job is the absolute job reference the worker launches; name is the job's display name."""
    item = {
        "id": uuid.uuid4().hex[:8], "job": job, "name": name or Path(job).stem, "run_name": run_name,
        "priority": int(priority),
        "status": "queued", "submitted": now_iso(), "notify_pane": notify_pane, "workdir": workdir,
        "resumes": 0, "run_id": None, "iter": None, "max_iters": None, "sec_it": None, "eta_h": None,
        "wandb_url": None, "last_iter": None, "last_change": None, "started": None, "reason": "",
        "tab_url": None,
    }

    def _add(data: dict) -> dict:
        data["items"].append(item)
        return item

    return update(_add, path)


def next_queued(data: dict, now: str) -> dict | None:
    """Highest priority first, then oldest. None while the queue is blocked (usage limit or no GPU)."""
    if data.get("blocked_until") and data["blocked_until"] > now:
        return None
    queued = [it for it in data["items"] if it["status"] == "queued"]
    if not queued:
        return None
    return min(queued, key=lambda it: (-it["priority"], it["submitted"]))


def running_count(data: dict) -> int:
    return sum(1 for it in data["items"] if it["status"] in ("running", "stalled"))


def cancel(item_id: str, path: Path | None = None) -> tuple[str, dict | None]:
    """Mark a queued item cancelled. Returns (previous status, item) or ('missing', None)."""

    def _cancel(data: dict) -> tuple[str, dict | None]:
        it = find(data, item_id)
        if it is None:
            return "missing", None
        prev = it["status"]
        if prev == "queued":
            it["status"] = "cancelled"
            it["reason"] = "cancelled before launch"
        return prev, dict(it)

    return update(_cancel, path)
