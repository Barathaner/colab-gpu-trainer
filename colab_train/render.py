"""Render a validated job spec into a self-contained Colab notebook (nbformat 4.5).

The notebook embeds the runner (colab_train/runner.py) and the job JSON. It contains Colab Secret
NAMES and public settings only. The user presses Run all; nothing else needs typing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUNNER_PATH = Path(__file__).parent / "runner.py"


def _cell_id(seed: str, index: int) -> str:
    return hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:12]


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def _md(seed: str, i: int, text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(seed, i), "metadata": {}, "source": _lines(text)}


def _code(seed: str, i: int, text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(seed, i), "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(text)}


def _raw_literal(text: str, what: str) -> str:
    """Embed text in a r'''...''' literal exactly. Refuse text that cannot round-trip."""
    if "'''" in text or text.endswith("\\"):
        raise ValueError(f"{what} cannot be embedded in a triple-quoted literal")
    return text


def secret_names(job: dict) -> list[tuple[str, str]]:
    """(env variable, Colab secret name) pairs the notebook must load. Names only."""
    pairs: list[tuple[str, str]] = []
    if job["repo"].get("token_secret"):
        pairs.append((job["repo"]["token_secret"], job["repo"]["token_secret"]))
    if job["repo"].get("fallback_token_secret"):
        pairs.append((job["repo"]["fallback_token_secret"], job["repo"]["fallback_token_secret"]))
    if job["wandb"]["mode"] == "online":
        pairs.append(("WANDB_API_KEY", job["wandb"]["api_key_secret"]))
    for env_name, secret in job["secret_env"].items():
        pairs.append((env_name, secret))
    seen: set[str] = set()
    out = []
    for p in pairs:
        if p[0] not in seen:
            seen.add(p[0])
            out.append(p)
    return out


def _wandb_url(job: dict) -> str:
    d = job["derived"]
    return f"https://wandb.ai/{d['wandb_entity']}/{d['wandb_project']}"


def _run_line(job: dict) -> str:
    """The run summary under the title: the W&B link only when W&B is on."""
    d = job["derived"]
    if job["wandb"]["mode"] == "disabled":
        return f"Run **{d['run_id']}** (no W&B: progress is printed in the cell and checkpoints are kept).\n\n"
    return (f"Run **{d['run_id']}** (W&B: [{d['wandb_entity']}/{d['wandb_project']}]({_wandb_url(job)})).\n\n")


def render_notebook(job: dict, runner_source: str | None = None) -> dict:
    runner = runner_source if runner_source is not None else RUNNER_PATH.read_text()
    _raw_literal(runner, "runner.py")
    seed = job["derived"]["run_id"]
    d = job["derived"]
    needs_drive = job["resume"]["source"] == "drive"
    pairs = secret_names(job)
    job_json = json.dumps(job, indent=2, ensure_ascii=False)
    _raw_literal(job_json, "job JSON")

    steps = [
        f"# colab-train: {job['name']}\n\n"
        f"{job.get('description') or ''}\n\n"
        f"{_run_line(job)}"
        "**Run all** does everything. Keep the tab open. A disconnect is normal: reconnect and "
        "**Run all** again; the run resumes from its newest checkpoint.\n\n"
        "One-time: Colab Secrets (key icon) with **Notebook access** ON for: "
        + ", ".join(f"`{s}`" for _, s in pairs) + ". Drive consent is asked only if the job uses Drive.",
        ("# Check the GPU and Python (no secrets here).\n"
         "import subprocess, sys\n"
         "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', "
         "'--format=csv'], capture_output=True, text=True).stdout)\n"
         "print('Python', sys.version.split()[0])\n"
         if job["runtime"].get("accelerator", "GPU") == "GPU" else
         "# CPU runtime (runtime.accelerator: none): no GPU check.\n"
         "import sys\n"
         "print('Python', sys.version.split()[0])\n"),
        "# Colab Secrets are read here. Values go into environment variables and are never printed.\n"
        "import os\n"
        "from google.colab import userdata\n\n"
        f"PAIRS = {json.dumps([list(p) for p in pairs], indent=4)}  # (env name, Colab secret name)\n\n"
        "for env_name, secret_name in PAIRS:\n"
        "    try:\n"
        "        value = userdata.get(secret_name)\n"
        "    except Exception as e:\n"
        "        raise RuntimeError(f'Colab secret {secret_name} is missing or Notebook access is off: {type(e).__name__}')\n"
        "    if not value:\n"
        "        raise RuntimeError(f'Colab secret {secret_name} is empty')\n"
        "    os.environ[env_name] = value\n"
        "    print(f'{env_name}: loaded')\n"
        + (
            "\n# This job resumes from Google Drive: mount it (Drive consent is asked once).\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n"
            "DRIVE_ROOT = '/content/drive/MyDrive/colab-train'\n"
            "os.makedirs(DRIVE_ROOT, exist_ok=True)\n"
            "print('Drive checkpoints:', DRIVE_ROOT)\n"
            if needs_drive else
            f"\n# No Drive: checkpoints live in {'W&B' if job['resume']['source'] == 'wandb' else 'this runtime only'}"
            f" (resume source {job['resume']['source']}).\n"
            "DRIVE_ROOT = None\n"
        ),
        "# Write the runner and the job spec (no secret values inside).\n"
        "import pathlib\n"
        "WORKDIR = pathlib.Path('/content/colab-train')\n"
        "WORKDIR.mkdir(parents=True, exist_ok=True)\n"
        f"(WORKDIR / 'runner.py').write_text(r'''{_raw_literal(runner, 'runner.py')}''')\n"
        f"(WORKDIR / 'job.json').write_text(r'''{job_json}''')\n"
        "print('runner and job written to', WORKDIR)\n",
        "# Training (and bench). Re-run this cell alone after a disconnect: it resumes.\n"
        "import subprocess, sys, pathlib\n"
        "WORKDIR = pathlib.Path('/content/colab-train')\n"
        "cmd = [sys.executable, str(WORKDIR / 'runner.py'), 'run', '--job', str(WORKDIR / 'job.json'),\n"
        "       '--workdir', str(WORKDIR)]\n"
        "if DRIVE_ROOT:\n"
        "    cmd += ['--drive-root', DRIVE_ROOT]\n"
        "# The runner's lines are echoed with print(): output written straight to the process's file descriptor is\n"
        "# not always shown in the cell, and a failure would then show only a traceback of this cell.\n"
        "# Every line also goes to outputs/runner.log. When the runner exits (also on failure), the outputs folder\n"
        "# (log, bench JSONs, the job's outputs: files) is packed and downloaded through the browser, so results\n"
        "# survive a failed upload and the runtime's deletion. colab-train's worker keeps the download.\n"
        "OUT = WORKDIR / 'outputs'\n"
        "OUT.mkdir(parents=True, exist_ok=True)\n"
        "rc = None\n"
        "try:\n"
        "    with open(OUT / 'runner.log', 'a') as log_file:\n"
        "        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "        for line in proc.stdout:\n"
        "            print(line, end='', flush=True)\n"
        "            log_file.write(line)\n"
        "        rc = proc.wait()\n"
        "finally:\n"
        "    try:\n"
        "        import shutil\n"
        "        from google.colab import files\n"
        f"        arc = shutil.make_archive('/content/colab-train-results-{d['run_id']}', 'gztar', OUT)\n"
        "        files.download(arc)\n"
        "        print('[colab-train] results downloaded:', pathlib.Path(arc).name, flush=True)\n"
        "    except Exception as e:  # the run's own result must still show\n"
        "        print('[colab-train] results download failed:', repr(e), flush=True)\n"
        "if rc:\n"
        "    raise RuntimeError(f'runner exited with status {rc}; the [colab-train] lines above say why')\n",
        "When the last cell ends with `Done.`, the run is finished or paused at its newest checkpoint. "
        + ("The curves and videos are on W&B. " if job["wandb"]["mode"] != "disabled" else "")
        + "Then **Runtime > Disconnect and delete runtime** to free the GPU.",
    ]
    cells = [_md(seed, 0, steps[0])]
    cells += [_code(seed, i, text) for i, text in enumerate(steps[1:5], start=1)]
    cells += [_md(seed, 5, steps[5])]
    runtime = job.get("runtime", {})
    metadata = {
        "accelerator": runtime.get("accelerator", "GPU"),
        "colab": {"provenance": [], **({"gpuType": runtime.get("gpu_type", "T4")}
                                       if runtime.get("accelerator", "GPU") == "GPU" else {})},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab_train": {"job": job["name"], "run_id": d["run_id"]},
    }
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": cells}


def write_notebook(job: dict, path: Path) -> Path:
    nb = render_notebook(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return path
