"""W&B side of `status`, `logs` and `list`. Reads only; uses the wandb library (key from ~/.netrc or
WANDB_API_KEY). The key is never printed. wandb is imported lazily so validate/render need no network."""
from __future__ import annotations

import itertools
import re
import tempfile
from pathlib import Path
from typing import Any

from .spec import SpecError

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# The exact reinstall command, shown wherever the optional wandb extra is missing.
INSTALL_HINT = ("uv tool install --force 'colab-train[wandb,browser] @ "
                "git+https://github.com/Barathaner/colab-gpu-trainer.git'  (pip: pip install 'colab-train[wandb]')")


def installed() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def _api():
    if not installed():
        raise SpecError([f"wandb is not installed; this job logs to W&B, so status, logs and the worker need it. "
                         f"Install it with: {INSTALL_HINT}"])
    import wandb
    return wandb.Api(timeout=60)


def project_path(job: dict) -> str:
    d = job["derived"]
    return f"{d['wandb_entity']}/{d['wandb_project']}"


def find_run(job: dict, api: Any = None, run_id: str | None = None):
    """Newest run whose name ends with the run id (the runner names runs exactly run_id; the
    training framework may prefix a timestamp). Bench runs are skipped."""
    api = api or _api()
    rid = run_id or job["derived"]["run_id"]
    runs = itertools.islice(api.runs(project_path(job), order="-created_at", per_page=100), 400)
    for r in runs:
        if r.name.endswith(rid) and not r.name.endswith("-bench"):
            return r
    return None


def _iter_of(job: dict, name: str) -> int | None:
    m = re.fullmatch(job["resume"]["checkpoint_regex"], Path(name).name)
    return int(m.group(1)) if m and m.groups() else None


def summarize(job: dict, run: Any) -> dict:
    key = job["wandb"]["iteration_key"]
    summary = run.summary
    iteration = summary.get(key)
    rows = []
    try:
        rows = list(run.history(keys=[key, "_runtime"], pandas=False, samples=500))
    except Exception:
        rows = []
    pts = sorted((r[key], r["_runtime"]) for r in rows
                 if isinstance(r.get(key), (int, float)) and isinstance(r.get("_runtime"), (int, float)))
    sec_per_iter = None
    if len(pts) >= 2:
        recent = pts[-20:]
        d_it = recent[-1][0] - recent[0][0]
        if d_it > 0:
            sec_per_iter = (recent[-1][1] - recent[0][1]) / d_it
    files = [(f.name, f.size) for f in run.files(per_page=1000)]
    ckpts = [it for n, _ in files if (it := _iter_of(job, n)) is not None]
    videos = sorted(n for n, _ in files if n.lower().endswith((".mp4", ".gif", ".webm")))
    max_iters = job["train"]["max_iters"]
    eta_h = None
    if run.state == "running" and sec_per_iter and isinstance(iteration, (int, float)) \
            and iteration < max_iters:
        eta_h = (max_iters - iteration) * sec_per_iter / 3600
    return {
        "run": run.name, "id": run.id, "state": run.state, "url": run.url,
        "iteration": iteration, "max_iters": max_iters,
        "sec_per_iter": round(sec_per_iter, 2) if sec_per_iter else None,
        "eta_hours": round(eta_h, 2) if eta_h is not None else None,
        "runtime_h": round((summary.get("_runtime") or 0) / 3600, 2),
        "newest_checkpoint": max(ckpts) if ckpts else None, "checkpoints": len(ckpts),
        "videos": len(videos), "latest_video": videos[-1] if videos else None,
        "video_logged": "video" in summary.keys(),
    }


def read_log(job: dict, run: Any, lines: int = 40) -> str:
    names = [f.name for f in run.files(per_page=1000)]
    if "output.log" not in names:
        return "(no output.log on this W&B run yet)"
    with tempfile.TemporaryDirectory(prefix="colab-train-log-") as tmp:
        fh = run.file("output.log").download(root=tmp, replace=True)
        path = Path(fh.name)
        fh.close()
        text = path.read_text(errors="replace")
    text = ANSI_RE.sub("", text)
    return "\n".join(text.splitlines()[-lines:])


def list_runs(job: dict, api: Any = None, limit: int = 10) -> list[dict]:
    api = api or _api()
    runs = itertools.islice(api.runs(project_path(job), order="-created_at", per_page=50), limit)
    return [{"name": r.name, "state": r.state, "created": str(r.created_at), "url": r.url} for r in runs]


def run_names(job: dict, api: Any = None, limit: int = 400) -> list[str]:
    """Display names of the newest runs in the job's W&B project (used to pick a run name nobody has used)."""
    api = api or _api()
    runs = itertools.islice(api.runs(project_path(job), order="-created_at", per_page=100), limit)
    return [r.name for r in runs]
