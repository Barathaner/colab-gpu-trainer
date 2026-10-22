#!/usr/bin/env python3
"""colab-train runner: clone, install, train with resume, bench. Runs ON the Colab runtime.

This file is embedded verbatim in every rendered notebook, so it must stay standard library only
(W&B is imported lazily and only for W&B resume). It reads a job JSON written by the notebook.
The job JSON holds Colab Secret NAMES only; secret VALUES arrive as environment variables set by
the notebook, and they are never printed or put on a command line.

Usage:
  python runner.py run  --job job.json --workdir /content/colab-train [--drive-root DIR]
  python runner.py plan --job job.json            (prints the train command, no network)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from urllib.parse import urlparse
import time
from dataclasses import dataclass
from pathlib import Path

LOG = "[colab-train]"
SYNC_EVERY_S = 60
POLL_S = 5
PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def log(msg: str) -> None:
    print(f"{LOG} {msg}", flush=True)


def expand(template: str, ctx: dict, where: str = "template") -> str:
    def rep(m: re.Match) -> str:
        key = m.group(1)
        if key not in ctx:
            raise KeyError(f"{where}: placeholder {{{key}}} has no value at this step")
        return str(ctx[key])

    return PLACEHOLDER_RE.sub(rep, template)


def _run(args: list[str], cwd: Path | None = None, env: dict | None = None, check: bool = True):
    # Only argv is logged. Secrets never appear in argv: git gets the token from an env var.
    log("$ " + " ".join(shlex.quote(a) for a in args))
    return subprocess.run(args, cwd=cwd, env=env, check=check)


# ---------------------------------------------------------------- context and commands

def base_ctx(job: dict, repo: Path) -> dict:
    d, t, v = job["derived"], job["train"], job["video"]
    return {
        "task": t["task"], "num_envs": t["num_envs"], "max_iters": t["max_iters"],
        "save_every": t["save_every"], "run_id": d["run_id"], "run_name": job["run"]["name"],
        "wandb_entity": d["wandb_entity"], "wandb_project": d["wandb_project"],
        "repo_dir": str(repo), "video_every": v["every"],
        "video_interval": v["every"] * v["steps_per_iter"], "video_length": v["length"],
    }


def train_context(job: dict, repo: Path, resume: "Resume | None") -> dict:
    ctx = base_ctx(job, repo)
    t = job["train"]
    if resume is None:
        ctx["iters"] = t["max_iters"]
        ctx["resume_args"] = ""
    else:
        # Frameworks such as rsl_rl restart their loop at the loaded iteration and run
        # num_learning_iterations more, so iters_mode "remaining" = max_iters - start.
        if job["resume"]["iters_mode"] == "remaining":
            ctx["iters"] = max(t["max_iters"] - resume.iteration, 1)
        else:
            ctx["iters"] = t["max_iters"]
        args = expand(job["resume"]["args"], {
            "resume_run": resume.run_name, "resume_checkpoint": resume.checkpoint,
            "resume_path": str(resume.path), "resume_iter": resume.iteration,
        }, "resume.args")
        ctx["resume_args"] = " " + args if args else ""
    flags = job["video"]["flags"] if ctx["video_every"] > 0 else ""
    ctx["video_flags"] = expand(flags, ctx, "video.flags") if flags else ""
    return ctx


# rsl_rl (and similar loops) write their training config to W&B when training starts. On a resumed run it differs from the
# first run (resume, load_run, max_iterations), and W&B refuses the change ("allow_val_change"). Resuming is
# intended, so resume.wandb_config_shim lets that one update through. Loaded only for resumed runs, via PYTHONPATH.
WANDB_RESUME_SHIM = """\
\"\"\"colab-train: allow the config change a training loop makes on a resumed W&B run (see runner.py).\"\"\"
try:
    import wandb.sdk.wandb_config as _wc
    _orig_update = _wc.Config.update

    def _update(self, d=None, allow_val_change=None):
        return _orig_update(self, d, allow_val_change=True if allow_val_change is None else allow_val_change)

    _wc.Config.update = _update
except Exception:
    pass
"""


def write_resume_shim(workdir: Path) -> Path:
    d = workdir / "wandb-resume-shim"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sitecustomize.py").write_text(WANDB_RESUME_SHIM)
    return d


def build_train_command(job: dict, repo: Path, resume: "Resume | None") -> list[str]:
    ctx = train_context(job, repo, resume)
    return shlex.split(expand(job["train"]["command"], ctx, "train.command"))


def build_install_commands(job: dict, repo: Path) -> list[list[str]]:
    ctx = base_ctx(job, repo)
    return [shlex.split(expand(c, ctx, f"install[{i}]")) for i, c in enumerate(job["install"])]


def build_bench_command(job: dict, repo: Path, checkpoint: Path, json_path: Path) -> list[str]:
    ctx = {**base_ctx(job, repo), "checkpoint": str(checkpoint), "bench_json": str(json_path)}
    return shlex.split(expand(job["bench"]["command"], ctx, "bench.command"))


# ---------------------------------------------------------------- git

def git_prefix(job: dict, source: str = "primary", host: str | None = None) -> list[str]:
    """git with the credential helper for the primary repo, or for the fallback repo (source='fallback').
    With a host, the helper is offered only for that host, so a token never goes to another server."""
    if source == "fallback":
        secret = job["repo"].get("fallback_token_secret")
        user = job["repo"].get("fallback_token_user") or "x-access-token"
        var = "COLAB_TRAIN_GIT_TOKEN_FALLBACK"
    else:
        secret, user, var = job["repo"].get("token_secret"), job["repo"]["token_user"], "COLAB_TRAIN_GIT_TOKEN"
    if not secret:
        return ["git"]
    # The helper reads the token from the environment at git run time: no token in argv or logs.
    helper = f'!f() {{ echo username={user}; echo "password=${var}"; }}; f'
    key = f"credential.https://{host}.helper" if host else "credential.helper"
    return ["git", "-c", f"{key}={helper}"]


def _sync_repo(job: dict, workdir: Path, source: str, url: str, branch: str) -> Path:
    repo = workdir / job["derived"]["repo_dir_name"]
    g = git_prefix(job, source, host=urlparse(url).hostname)
    if (repo / ".git").is_dir():
        _run(["git", "-C", str(repo), "remote", "set-url", "origin", url])
        _run([*g, "-C", str(repo), "fetch", "origin", branch])
        _run(["git", "-C", str(repo), "checkout", "-q", "-B", branch, f"origin/{branch}"])
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        _run([*g, "clone", "--branch", branch, url, str(repo)])
    if job["repo"]["submodules"]:
        _run([*g, "-C", str(repo), "submodule", "update", "--init", "--recursive"], check=False)
    return repo


def clone_or_update(job: dict, workdir: Path) -> Path:
    """The primary repo. If it cannot be cloned or fetched and repo.fallback_url is set (e.g. the primary host is down),
    the same branch is taken from the fallback URL (e.g. GitHub). Only that second attempt is made."""
    branch = job["repo"]["branch"]
    try:
        return _sync_repo(job, workdir, "primary", job["repo"]["url"], branch)
    except (subprocess.CalledProcessError, OSError) as e:
        fb = job["repo"].get("fallback_url")
        if not fb:
            raise
        log(f"primary repo failed ({type(e).__name__}); using the fallback repo")
        fb_branch = job["repo"].get("fallback_branch") or branch
        return _sync_repo(job, workdir, "fallback", fb, fb_branch)


# ---------------------------------------------------------------- environment

def build_env(job: dict) -> dict:
    env = dict(os.environ)
    if job.get("python"):
        env["UV_PYTHON"] = str(job["python"])
        env["UV_PYTHON_PREFERENCE"] = "only-managed"
    if job["wandb"]["mode"] != "disabled":
        d = job["derived"]
        env.update({
            "WANDB_MODE": job["wandb"]["mode"], "WANDB_ENTITY": d["wandb_entity"],
            "WANDB_PROJECT": d["wandb_project"], "WANDB_RUN_ID": d["run_id"],
            "WANDB_NAME": d["run_id"], "WANDB_RESUME": "allow",
        })
    else:  # a trainer that calls wandb.init anyway (rsl_rl, mjlab) then runs without asking for a key
        env["WANDB_MODE"] = "disabled"
    env.update({k: str(v) for k, v in job["env"].items()})
    return env


def install(job: dict, repo: Path, env: dict) -> None:
    cmds = build_install_commands(job, repo)
    if job.get("python"):
        if shutil.which("uv") is None:
            _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        _run(["uv", "python", "install", str(job["python"])], env=env)
    elif any(c and c[0] == "uv" for c in cmds) and shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    for cmd in cmds:
        _run(cmd, cwd=repo, env=env)


def _repo_python(repo: Path) -> list[str]:
    """The interpreter the install steps used: uv's locked env for a uv project, else the runtime's python."""
    if (repo / "uv.lock").exists() and shutil.which("uv"):
        return ["uv", "run", "--frozen", "python"]
    return [sys.executable]


def _can_render(repo: Path, backend: str, env: dict) -> bool:
    e = {**env, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}
    code = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
            "mujoco.Renderer(m, 64, 64).render()")
    r = subprocess.run([*_repo_python(repo), "-c", code], cwd=repo, env=e, capture_output=True)
    return r.returncode == 0


def pick_gl_backend(repo: Path, env: dict) -> str:
    """egl (GPU), then osmesa; install the GL system libraries once if neither works."""
    for backend in ("egl", "osmesa"):
        if _can_render(repo, backend, env):
            return backend
    if shutil.which("apt-get"):
        _run(["apt-get", "update", "-qq"], check=False)
        _run(["apt-get", "install", "-y", "-qq", "libegl1", "libgl1", "libosmesa6"], check=False)
    for backend in ("egl", "osmesa"):
        if _can_render(repo, backend, env):
            return backend
    return "disable"


# ---------------------------------------------------------------- checkpoints and resume

@dataclass
class Resume:
    run_name: str
    checkpoint: str
    iteration: int
    path: Path


def _iteration(job: dict, name: str) -> int | None:
    m = re.fullmatch(job["resume"]["checkpoint_regex"], name)
    if not m:
        return None
    return int(m.group(1)) if m.groups() else 0


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def newest_local_checkpoint(job: dict, repo: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for p in repo.glob(job["resume"]["checkpoint_glob"]):
        if not p.is_file():
            continue
        it = _iteration(job, p.name)
        if it is not None and (best is None or it > best[0]):
            best = (it, p)
    return best[1] if best else None


def sync_up(job: dict, repo: Path, drive_run: Path) -> list[str]:
    """Copy checkpoints and sync_extra files to Drive (same relative paths). Only complete files."""
    copied: list[str] = []
    patterns = [job["resume"]["checkpoint_glob"], *job["resume"]["sync_extra"]]
    for pat in patterns:
        for src in repo.glob(pat):
            if not src.is_file():
                continue
            rel = src.relative_to(repo)
            dst = drive_run / rel
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            _atomic_copy(src, dst)
            copied.append(str(rel))
    return copied


def restore_drive(job: dict, repo: Path, drive_run: Path) -> Resume | None:
    best: tuple[int, Path] | None = None
    for src in drive_run.rglob("*"):
        if not src.is_file() or src.name.endswith(".tmp"):
            continue
        it = _iteration(job, src.name)
        if it is not None and (best is None or it > best[0]):
            best = (it, src)
    if best is None:
        return None
    it, newest = best
    for src in drive_run.rglob("*"):
        if not src.is_file() or src.name.endswith(".tmp"):
            continue
        dst = repo / src.relative_to(drive_run)
        if not dst.exists():
            _atomic_copy(src, dst)
    local = repo / newest.relative_to(drive_run)
    return Resume(run_name=newest.parent.name, checkpoint=newest.name, iteration=it, path=local)


def restore_wandb(job: dict, repo: Path) -> Resume | None:
    """Download the newest checkpoint file of the newest W&B run whose name ends with run_id."""
    import wandb  # lazy: only needed for W&B resume

    d = job["derived"]
    try:
        runs = list(wandb.Api(timeout=60).runs(f"{d['wandb_entity']}/{d['wandb_project']}",
                                               order="-created_at", per_page=50))
    except Exception as e:  # no network, no such project: start fresh
        log(f"W&B resume lookup failed ({type(e).__name__}); starting fresh.")
        return None
    run = next((r for r in runs if r.name.endswith(d["run_id"]) and not r.name.endswith("-bench")), None)
    if run is None:
        return None
    ckpts = []
    for f in run.files(per_page=1000):
        it = _iteration(job, Path(f.name).name)
        if it is not None:
            ckpts.append((it, f))
    if not ckpts:
        return None
    iteration, f = max(ckpts, key=lambda c: c[0])
    key = job["resume"].get("experiment_config_key")
    experiment = (Path(str(run.config.get(key, ""))).parent.name if key else "") \
        or d["wandb_project"].replace("-", "_")
    dst_dir = repo / expand(job["resume"]["restore_dir"],
                            {"run_name": run.name, "experiment": experiment, "run_id": d["run_id"]},
                            "resume.restore_dir")
    dst_dir.mkdir(parents=True, exist_ok=True)
    fh = f.download(root=str(dst_dir), replace=True)
    local = Path(fh.name).resolve()
    fh.close()
    return Resume(run_name=run.name, checkpoint=local.name, iteration=iteration, path=local)


# ---------------------------------------------------------------- progress (no W&B needed)

_PROGRESS_CACHE: dict[str, re.Pattern] = {}


def parse_progress(job: dict, line: str) -> tuple[int, int | None] | None:
    """(iteration, total or None) when a training stdout line matches progress.regex, else None.
    The iteration is the named group "iter" or the first group; "total" is an optional named group."""
    pattern = (job.get("progress") or {}).get("regex")
    if not pattern:
        return None
    rx = _PROGRESS_CACHE.get(pattern) or _PROGRESS_CACHE.setdefault(pattern, re.compile(pattern))
    m = rx.search(line)
    if not m:
        return None
    gd = m.groupdict()
    try:
        it = int(gd["iter"]) if gd.get("iter") is not None else int(m.group(1))
    except (TypeError, ValueError, IndexError):
        return None
    total = int(gd["total"]) if (gd.get("total") or "").isdigit() else None
    return it, total


def _pump(stream, job: dict, seen: dict) -> None:
    """Echo every training line and log a progress line when the iteration changes."""
    for raw in stream:
        line = raw.rstrip("\n")
        print(line, flush=True)
        hit = parse_progress(job, line)
        if hit is None or hit[0] == seen.get("iter"):
            continue
        it, total = hit
        total = total or job["train"]["max_iters"]
        seen["iter"] = it
        pct = f" ({100 * it // total}%)" if total else ""
        log(f"progress: {it}/{total}{pct}")


# ---------------------------------------------------------------- train and bench

def train(job: dict, repo: Path, cmd: list[str], env: dict, drive_run: Path | None) -> int:
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    budget = job["limits"]["max_hours"] * 3600
    # stdout and stderr go through one pipe; a reader thread echoes lines and parses progress while this loop
    # syncs checkpoints and watches the time budget.
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")
    seen: dict = {}
    reader = threading.Thread(target=_pump, args=(proc.stdout, job, seen), daemon=True)
    reader.start()
    t0 = time.time()
    last_sync = 0.0
    stopped = False
    while proc.poll() is None:
        if drive_run is not None and time.time() - last_sync >= SYNC_EVERY_S:
            copied = sync_up(job, repo, drive_run)
            if copied:
                log(f"Synced {len(copied)} file(s) to Drive.")
            last_sync = time.time()
        if not stopped and time.time() - t0 >= budget:
            log(f"Time budget of {job['limits']['max_hours']} h reached: sending SIGINT to training. "
                "The newest checkpoint is kept.")
            proc.send_signal(signal.SIGINT)
            stopped = True
        time.sleep(POLL_S)
    reader.join(timeout=30)
    if drive_run is not None:
        sync_up(job, repo, drive_run)
    if proc.returncode != 0 and not stopped:
        raise RuntimeError(f"training exited with {proc.returncode}; checkpoints so far are kept")
    return proc.returncode


def bench(job: dict, repo: Path, env: dict, drive_run: Path | None) -> None:
    if not job["bench"].get("command"):
        return
    ck = newest_local_checkpoint(job, repo)
    if ck is None:
        log("No checkpoint to benchmark.")
        return
    out = repo / "bench"
    out.mkdir(exist_ok=True)
    json_path = out / f"bench_{ck.stem}.json"
    cmd = build_bench_command(job, repo, ck, json_path)
    r = _run(cmd, cwd=repo, env=env, check=False)
    if json_path.exists() and drive_run is not None:
        drive_run.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, drive_run / json_path.name)
        log(f"Bench JSON saved to {drive_run / json_path.name}")
    if r.returncode != 0:
        raise RuntimeError(f"bench exited with {r.returncode}")


def run_job(job: dict, workdir: Path, drive_root: Path | None) -> None:
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        gpu_text = gpu.stdout.strip()
    except OSError:
        gpu_text = ""
    log(f"GPU: {gpu_text or 'none found'}")
    rt = job.get("runtime") or {}
    if not gpu_text and rt.get("accelerator", "GPU") == "GPU":
        # Colab gave a CPU-only runtime (no nvidia-smi). Training would run on the CPU and fail or crawl; stop here.
        raise RuntimeError(f"no GPU attached to this runtime (nvidia-smi not found): Runtime > Disconnect and delete "
                           f"runtime, then Run all again when Colab assigns a {rt.get('gpu_type', 'T4')}. "
                           "Set runtime.accelerator: none in the job if it does not need a GPU.")
    secret = job["repo"].get("token_secret")
    if secret:
        if not os.environ.get(secret):
            raise SystemExit(f"{LOG} environment variable {secret} is empty: add the Colab secret "
                             f"{secret} with Notebook access on, then run the setup cell again.")
        os.environ["COLAB_TRAIN_GIT_TOKEN"] = os.environ[secret]
    fb_secret = job["repo"].get("fallback_token_secret")
    if fb_secret and os.environ.get(fb_secret):
        os.environ["COLAB_TRAIN_GIT_TOKEN_FALLBACK"] = os.environ[fb_secret]
    if job["wandb"]["mode"] == "online" and not os.environ.get("WANDB_API_KEY") \
            and not Path.home().joinpath(".netrc").exists():
        raise SystemExit(f"{LOG} WANDB_API_KEY missing (Colab secret). Add it under the key icon with Notebook "
                         "access on, or set wandb.mode: disabled in the job to train without W&B.")

    repo = clone_or_update(job, workdir)
    env = build_env(job)
    if job["wandb"]["mode"] != "disabled":
        try:
            import wandb  # noqa: F401
        except ImportError:  # Colab does not always ship wandb; the job's W&B logging and resume need it
            _run([sys.executable, "-m", "pip", "install", "-q", "wandb"])
    install(job, repo, env)
    if job["mujoco_gl"]:
        backend = pick_gl_backend(repo, env)
        log(f"MUJOCO_GL={backend}" + ("" if backend != "disable" else " (videos turned off)"))
        env["MUJOCO_GL"] = env["PYOPENGL_PLATFORM"] = backend
        if backend == "disable":
            job["video"]["every"] = 0

    drive_run = None
    if job["resume"]["source"] == "drive":
        if drive_root is None:
            raise SystemExit(f"{LOG} resume.source is drive but no --drive-root was given")
        drive_run = Path(drive_root) / job["derived"]["run_id"]
        drive_run.mkdir(parents=True, exist_ok=True)
        resume = restore_drive(job, repo, drive_run)
        source = "Drive"
    elif job["resume"]["source"] == "wandb":
        resume = restore_wandb(job, repo)
        source = "W&B"
    else:
        resume, source = None, ""

    start = 0
    if resume is not None:
        start = resume.iteration
        log(f"Resuming {resume.run_name} from {resume.checkpoint} (iteration {start}, {source}).")
        if start >= job["train"]["max_iters"] - 1:
            log(f"Checkpoint {start} already reaches max_iters {job['train']['max_iters']}: nothing to train.")
            resume_done = True
        else:
            resume_done = False
    else:
        log("No checkpoint found: starting fresh.")
        resume_done = False
    wb_text = (f"W&B {job['derived']['wandb_entity']}/{job['derived']['wandb_project']}"
               if job["wandb"]["mode"] != "disabled" else "no W&B")
    log(f"Run {job['derived']['run_id']} ({wb_text}); repo {repo}")
    if not resume_done:
        cmd = build_train_command(job, repo, resume)
        if resume is not None and job["resume"].get("wandb_config_shim"):
            shim = write_resume_shim(workdir)
            env = {**env, "PYTHONPATH": str(shim) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")}
        train(job, repo, cmd, env, drive_run)
    bench(job, repo, env, drive_run)
    log("Done. Checkpoints are kept (Drive or W&B) and the run is on W&B." if job["wandb"]["mode"] != "disabled"
        else "Done. Checkpoints are kept (Drive or local storage, see resume.source).")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="colab-train-runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="clone, install, train with resume, bench")
    r.add_argument("--job", required=True, type=Path)
    r.add_argument("--workdir", type=Path, default=Path("/content/colab-train"))
    r.add_argument("--drive-root", type=Path, default=None)
    pl = sub.add_parser("plan", help="print the train command for a fresh start (no network)")
    pl.add_argument("--job", required=True, type=Path)
    pl.add_argument("--workdir", type=Path, default=Path("/content/colab-train"))
    a = p.parse_args(argv)
    job = json.loads(a.job.read_text())
    if a.cmd == "plan":
        repo = a.workdir / job["derived"]["repo_dir_name"]
        for c in build_install_commands(job, repo):
            print("install: " + " ".join(shlex.quote(x) for x in c))
        print("train:   " + " ".join(shlex.quote(x) for x in build_train_command(job, repo, None)))
        return
    run_job(job, a.workdir, a.drive_root)


if __name__ == "__main__":
    main()
