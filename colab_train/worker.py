"""Scheduler for the Colab queue: reconcile running jobs with W&B and the browser, launch the next queued job.

Pure functions (reconcile, decide, notify) take their inputs as values, so tests need no network. probe() and
launch_child() are the only parts that touch W&B, the browser or a child process; the worker loop uses them.

Rules (from the coordinator's brief):
- RUNNING while the W&B run exists and its iteration advances; FINISHED at max_iters or W&B finished.
- STALLED when the iteration has not moved for stall_min minutes (default 15).
- FAILED on a crashed/killed W&B run, a runner error line, or no run 45 min after launch. Jobs with W&B off are
  followed through the runner's progress and Done lines in their Colab tab.
- DISCONNECTED when the Colab tab or runtime is gone. Resumed as a fresh launch of the same run name (the runner
  restores the newest W&B checkpoint), at most max_resumes times (default 2). Never a keep-alive.
- A usage-limit or no-GPU stop blocks the queue for backoff_h hours (default 2). Captcha, sign-in and robot-check
  stops mark the item needs-human: nothing is clicked past.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import queue as q

EVENTS_FILE_ENV = "COLAB_TRAIN_EVENTS"
# Fatal lines only. "primary repo failed; using the fallback repo" and "W&B resume lookup failed; starting fresh"
# are recoveries, not failures.
ERROR_LOG = re.compile(r"training exited with|runner exited with status \d+|Traceback \(most recent call last\)|"
                       r"\[colab-train\] (?!.*(?:fallback|starting fresh)).*\b(failed|error)\b", re.I)
NO_GPU_OR_LIMIT = re.compile(r"usage limit|no GPU|GPU backend", re.I)
HUMAN_STOP = re.compile(r"captcha|robot check|sign-in|not signed in", re.I)
NO_RUN_TIMEOUT_MIN = 45
RUNNER_EXIT = re.compile(r"training exited with \d+|runner exited with status \d+")
SESSION_RETRY_MIN = 10  # Colab's one-session limit: try again later, the queued job is not failed


def settings(config: dict) -> dict:
    def num(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default
    return {
        "max_parallel": int(num("max_parallel", 1)),
        "stall_min": num("stall_min", 15),
        "max_resumes": int(num("max_resumes", 2)),
        "backoff_h": num("backoff_h", 2),
        "interval_s": num("interval_s", 120),
        "notify_cmd": config.get("notify_cmd") or None,
        "error_pane": config.get("error_pane") or None,
    }


def events_path() -> Path:
    override = os.environ.get(EVENTS_FILE_ENV)
    return Path(override) if override else q.queue_path().with_name("events.log")


def _ago_min(then: str | None, now: str) -> float:
    if not then:
        return 0.0
    t0 = dt.datetime.fromisoformat(then)
    t1 = dt.datetime.fromisoformat(now)
    return (t1 - t0).total_seconds() / 60.0


def reconcile(item: dict, probe: dict, now: str, cfg: dict) -> dict:
    """The changes for one running item, given what W&B and the browser showed. Returns a dict of fields to set.
    Always contains 'status' and 'event' (a human line, or '')."""
    st = cfg
    out: dict[str, Any] = {"event": ""}
    run_state = probe.get("run_state")
    it_now = probe.get("iteration")
    max_it = probe.get("max_iters") or item.get("max_iters")
    for k in ("iter", "max_iters", "sec_it", "eta_h", "wandb_url", "run_id"):
        if k in probe and probe[k] is not None:
            out[k] = probe[k]
    if max_it is not None:
        out["max_iters"] = max_it
    if it_now is not None:
        out["iter"] = it_now
        first = item.get("first_iter")
        if first is None:
            first = out["first_iter"] = it_now
        if max_it is not None and first > max_it:
            # a run that continues another run's iteration count (8997 -> +2500): max_iters counts from its start
            it_now = it_now - first

    if probe.get("log_error") or run_state in ("crashed", "failed", "killed"):
        why = probe.get("log_error") or f"W&B run {run_state}"
        return {**out, "status": "failed", "reason": why[:200],
                "event": f"{item['job']} {item['run_name'] or ''}: FAILED ({why[:120]})"}
    if run_state == "finished" or (it_now is not None and max_it is not None and it_now >= max_it):
        return {**out, "status": "finished", "reason": "max_iters reached or W&B finished",
                "event": f"{item['job']}: FINISHED at iteration {it_now}"}
    if run_state is None:
        if _ago_min(item.get("started"), now) > NO_RUN_TIMEOUT_MIN:
            return {**out, "status": "failed",
                    "reason": f"no W&B run or runner output {NO_RUN_TIMEOUT_MIN} min after launch",
                    "event": f"{item['job']}: FAILED (no W&B run or runner output after {NO_RUN_TIMEOUT_MIN} min)"}
        return {**out, "status": "running"}

    if probe.get("runtime_gone") is True:
        if int(item.get("resumes", 0)) < st["max_resumes"]:
            return {**out, "status": "queued", "resumes": int(item.get("resumes", 0)) + 1,
                    "reason": "disconnected: fresh launch from the last W&B checkpoint",
                    "event": f"{item['job']}: DISCONNECTED, re-queued to resume ({int(item.get('resumes', 0)) + 1}/"
                             f"{st['max_resumes']})"}
        return {**out, "status": "failed", "reason": "disconnected; resume limit reached",
                "event": f"{item['job']}: FAILED (disconnected, resume limit {st['max_resumes']} reached)"}

    # still running: track progress
    if it_now is not None and it_now != item.get("last_iter"):
        out["last_iter"] = it_now
        out["last_change"] = now
        if item.get("status") == "stalled":
            out["status"] = "running"
            out["event"] = f"{item['job']}: running again at iteration {it_now}"
            return out
    elif _ago_min(item.get("last_change") or item.get("started"), now) > st["stall_min"]:
        if item.get("status") != "stalled":
            out["status"] = "stalled"
            out["event"] = (f"{item['job']}: STALLED, no new iteration for {int(st['stall_min'])} min "
                            f"(iteration {it_now})")
            return out
    out["status"] = item.get("status", "running") if item.get("status") == "stalled" else "running"
    return out


def decide_launch(data: dict, cfg: dict, now: str) -> dict | None:
    """The next queued item to launch, or None when the slot is full or the queue is blocked."""
    if q.running_count(data) >= cfg["max_parallel"]:
        return None
    return q.next_queued(data, now)


def classify_launch(rc: int, output: str) -> tuple[str, str]:
    """(status, reason) for a finished launch child process."""
    if rc == 0:
        return "running", ""
    stop = re.search(r"STOP: ([^\n]*)", output)
    if stop:
        reason = stop.group(1).split(". ")[0].strip()
        if re.search(r"too many sessions", reason, re.I):
            return "queued-session", reason
        if NO_GPU_OR_LIMIT.search(reason):
            return "queued-blocked", reason
        if HUMAN_STOP.search(reason):
            return "needs-human", reason
    if re.search(r"refused:", output):
        # a local count (max_parallel or a stale runtime record), not Colab's quota: retry soon, not after backoff_h
        return "queued-session", "refused by the local guard (runtimes recorded >= max_parallel)"
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    # a multi-line error message ends with its screenshot path; its first line names the error
    err = next((ln for ln in lines if re.match(r"(error|[A-Za-z]*Error)\b", ln)), None)
    return "failed", (err or (lines[-1] if lines else ""))[:200]


def notify(item: dict, message: str, cfg: dict, now: str | None = None) -> None:
    """Append one line to events.log and run the notify hook, if configured. Never raises.
    Failures and human-stop events go to cfg['error_pane'] (the session that fixes errors), else to the job's pane."""
    line = f"{now or now_iso()} {item.get('id', '-')} {message}"
    if item.get("status") in ("failed", "needs-human") and cfg.get("error_pane"):
        item = {**item, "notify_pane": cfg["error_pane"]}
    try:
        path = events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    cmd = cfg.get("notify_cmd")
    if cmd:
        pane = item.get("notify_pane") or ""
        argv = [a.replace("{pane}", pane) for a in shlex.split(cmd)]
        try:
            subprocess.run([*argv, line], timeout=30, check=False, capture_output=True)
        except (OSError, subprocess.SubprocessError):
            pass


def now_iso() -> str:
    return q.now_iso()


# ---------------------------------------------------------------- real probes (network, browser)

PROGRESS_LINE = re.compile(r"^\[colab-train\] progress: (\d+)/(\d+)")
DONE_LINE = re.compile(r"^\[colab-train\] Done\.")


def parse_tab_lines(lines: list[str]) -> dict:
    """Progress from the runner's own lines in the Colab tab. Used when the job has no W&B run to read."""
    out: dict[str, Any] = {}
    for ln in lines:
        m = PROGRESS_LINE.match(ln)
        if m:
            out["iteration"], out["max_iters"] = int(m.group(1)), int(m.group(2))
    if any(DONE_LINE.match(ln) for ln in lines):
        out["run_state"] = "finished"
    elif lines:
        out["run_state"] = "running"
    err = next((ERROR_LOG.search(ln) for ln in lines if ERROR_LOG.search(ln)), None)
    if err:
        out["log_error"] = err.group(0)
    return out


def early_exit_from_tab(lines: list[str]) -> dict:
    """A W&B job that died before its W&B run started (install, data or env setup) shows that only in its tab."""
    err = next((RUNNER_EXIT.search(ln) for ln in lines if RUNNER_EXIT.search(ln)), None)
    return {"log_error": err.group(0)} if err else {}


def saved_tab(workdir: str | Path | None, job_name: str) -> tuple[str | None, str]:
    """(tab URL, notebook stem) that `launch --browser` saved in <workdir>/.colab-train/state.json."""
    path = Path(workdir or Path.cwd()) / ".colab-train" / "state.json"
    try:
        entry = json.loads(path.read_text()).get(job_name) or {}
    except (OSError, ValueError, AttributeError):
        return None, ""
    return (entry.get("browser") or {}).get("url"), Path(entry.get("notebook") or "").stem


def probe(item: dict, job_ref: str, run_name: str | None, workdir: str | None) -> dict:
    """What W&B and the browser show for one running item. Missing pieces are None, never a guess."""
    out: dict[str, Any] = {}
    cwd = Path(workdir) if workdir else Path.cwd()
    job_name, wandb_on = None, True
    try:
        from . import spec, wandb_status
        old = Path.cwd()
        os.chdir(cwd)
        try:
            job = spec.load(spec.resolve_job_arg(job_ref), run_name=run_name)
            job_name, wandb_on = job["name"], job["wandb"]["mode"] != "disabled"
            run = wandb_status.find_run(job) if wandb_on else None
            if run is not None:
                s = wandb_status.summarize(job, run)
                out.update({"run_state": s.get("state"), "iteration": s.get("iteration"),
                            "max_iters": s.get("max_iters"), "sec_it": s.get("sec_per_iter"),
                            "eta_h": s.get("eta_hours"), "wandb_url": s.get("url"),
                            "run_id": job["derived"]["run_id"]})
                if run.state == "running":
                    text = wandb_status.read_log(job, run, lines=60)
                    m = ERROR_LOG.search(text)
                    if m:
                        out["log_error"] = m.group(0)
            else:
                out["run_state"] = None
        finally:
            os.chdir(old)
    except Exception as e:  # noqa: BLE001 - a probe failure is "unknown", not a verdict
        out["probe_error"] = f"{type(e).__name__}"
    url, stem = saved_tab(cwd, job_name) if job_name else (None, "")
    gone, lines = browser_tab(item.get("tab_url") or url, stem)
    out["runtime_gone"] = gone
    if not wandb_on and lines is not None:
        out.update(parse_tab_lines(lines))
    elif lines is not None and out.get("run_state") is None:
        out.update(early_exit_from_tab(lines))
    return out


def browser_tab(url: str | None, stem: str) -> tuple[bool | None, list[str] | None]:
    """(runtime gone, runner lines) for the job's Colab tab. Gone = tab missing or a Reconnect button.
    (None, None) when the browser is not reachable or the tab is unknown: unknown, do not act on it."""
    if not url and not stem:
        return None, None
    try:
        from . import browser, colab_web
        with browser.cdp_context() as ctx:
            page = colab_web.find_job_page(ctx, url, stem)
            if page is None:
                return True, None
            gone = page.get_by_role("button", name=re.compile(r"^Reconnect\b")).count() > 0
            text = colab_web.page_text(page).splitlines()
            return gone, [ln.rstrip() for ln in text if ln.startswith("[colab-train]") or ERROR_LOG.search(ln)]
    except Exception:  # noqa: BLE001 - no browser: unknown, do not act on it
        return None, None


def browser_runtime_gone(item: dict) -> bool | None:
    """True when the job's Colab tab is missing or shows a Reconnect button; None when unknown."""
    return browser_tab(item.get("tab_url"), "")[0]


def launch_child(item: dict, cfg_env: dict) -> tuple[int, str]:
    """Run `colab-train launch JOB --browser --run-name N` as a foreground child and return (rc, output)."""
    argv = [sys.executable, "-m", "colab_train", "launch", item["job"], "--browser"]
    if item.get("run_name"):
        argv += ["--run-name", item["run_name"]]
    proc = subprocess.run(argv, cwd=item.get("workdir") or None, capture_output=True, text=True,
                          env={**os.environ, **cfg_env}, timeout=3600)
    return proc.returncode, proc.stdout + proc.stderr


def stop_child(item: dict, cfg_env: dict) -> tuple[int, str]:
    """Run `colab-train stop JOB --browser --yes` as a foreground child and return (rc, output)."""
    argv = [sys.executable, "-m", "colab_train", "stop", item["job"], "--browser", "--yes"]
    if item.get("run_name"):
        argv += ["--run-name", item["run_name"]]
    proc = subprocess.run(argv, cwd=item.get("workdir") or None, capture_output=True, text=True,
                          env={**os.environ, **cfg_env}, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def pythonpath_env() -> dict:
    root = str(Path(__file__).resolve().parent.parent)
    return {"PYTHONPATH": root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}


# ---------------------------------------------------------------- one pass and the loop

def run_once(cfg: dict, probe_fn: Callable[[dict], dict] | None = None,
             launcher: Callable[[dict], tuple[int, str]] | None = None, now: str | None = None,
             stopper: Callable[[dict], tuple[int, str]] | None = None) -> list[str]:
    """One reconcile pass plus at most one launch. Returns the event lines it wrote."""
    config_probe = probe_fn or _default_probe
    now = now or now_iso()
    events: list[str] = []
    snap = q.snapshot()
    for it in [i for i in snap["items"] if i["status"] in ("running", "stalled")]:
        pr = config_probe(it)
        patch: dict[str, Any] = {}

        def _apply(data: dict, it_id: str = it["id"], pr: dict = pr) -> dict | None:
            cur = q.find(data, it_id)
            if cur is None or cur["status"] not in ("running", "stalled"):
                return None
            ch = reconcile(cur, pr, now, cfg)
            ev = ch.pop("event", "")
            cur.update({k: v for k, v in ch.items()})
            if cur["status"] in ("queued",):
                cur["started"] = None
            return {"event": ev, "item": dict(cur)} if ev else None

        res = q.update(_apply)
        if res:
            events.append(res["event"])
            notify(res["item"], res["event"], cfg, now)
            if res["item"]["status"] == "failed" and stopper is not None:
                # a failed run's runtime would sit idle and use compute units until Colab ends it
                rc, out = stopper(res["item"])
                events.append(f"{res['item']['job']}: runtime stop after failure "
                              f"{'done' if rc == 0 else 'FAILED, stop it by hand'}")
    launch_next = decide_launch(q.snapshot(), cfg, now)
    if launch_next and launcher is not None:
        cur_id = launch_next["id"]

        def _claim(data: dict) -> dict | None:
            cur = q.find(data, cur_id)
            if cur is None or cur["status"] != "queued":
                return None
            cur["status"] = "launching"
            cur["started"] = now
            return dict(cur)

        claimed = q.update(_claim)
        if claimed:
            rc, output = launcher(claimed)
            if rc != 0:
                try:  # the whole launch output; the queue keeps only one line of it
                    events_path().with_name(f"launch-{cur_id}.log").write_text(output)
                except OSError:
                    pass

            def _settle(data: dict) -> dict:
                cur = q.find(data, cur_id)
                status, reason = classify_launch(rc, output)
                if status == "queued-session":
                    cur["status"] = "queued"
                    data["blocked_until"] = (dt.datetime.fromisoformat(now) +
                                             dt.timedelta(minutes=SESSION_RETRY_MIN)).isoformat(timespec="seconds")
                    ev = f"{cur['job']}: QUEUED again ({reason}; retry in {SESSION_RETRY_MIN} min)"
                elif status == "queued-blocked":
                    cur["status"] = "queued"
                    data["blocked_until"] = (dt.datetime.fromisoformat(now) +
                                             dt.timedelta(hours=cfg["backoff_h"])).isoformat(timespec="seconds")
                    ev = f"{cur['job']}: BLOCKED ({reason}); queue paused {cfg['backoff_h']} h"
                elif status == "needs-human":
                    cur["status"] = "needs-human"
                    ev = f"{cur['job']}: NEEDS A HUMAN ({reason}). Nothing was clicked past it."
                elif status == "failed":
                    cur["status"] = "failed"
                    ev = f"{cur['job']}: FAILED to launch ({reason})"
                else:
                    cur["status"] = "running"
                    ev = f"{cur['job']}: LAUNCHED on Colab (run {cur.get('run_name') or '-'})"
                cur["reason"] = reason
                cur["last_change"] = now
                return {"event": ev, "item": dict(cur)}

            res = q.update(_settle)
            events.append(res["event"])
            notify(res["item"], res["event"], cfg, now)
    return events


def _default_probe(item: dict) -> dict:
    return probe(item, item["job"], item.get("run_name"), item.get("workdir"))


def loop(cfg: dict, interval_s: float, probe_fn: Callable[[dict], dict] | None = None,
         launcher: Callable[[dict], tuple[int, str]] | None = None, once: bool = False,
         stopper: Callable[[dict], tuple[int, str]] | None = None) -> None:
    """Foreground loop. Ctrl+C stops it; nothing is detached."""
    while True:
        for line in run_once(cfg, probe_fn, launcher, stopper=stopper):
            print(line, flush=True)
        if once:
            return
        time.sleep(interval_s)
