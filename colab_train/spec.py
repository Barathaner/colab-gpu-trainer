"""Job spec: YAML -> validated, normalized dict used by the CLI, the notebook renderer and the runner.

A job spec holds NO secret values, only Colab Secret names (repo.token_secret, wandb.api_key_secret,
secret_env). Placeholders such as {task} or {iters} are expanded at run time by the runner.
"""
from __future__ import annotations

import copy
import difflib
import re
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).parent / "presets"

SECRET_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
SECRET_LIKE_KEY_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|CREDENTIAL)", re.I)
SECRET_LIKE_VALUE_RE = re.compile(r"^(ghp_|gho_|ghs_|github_pat_|glpat-|sk-|wandb_v1_|xox[bp]-)")
PLAIN_LONG_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/_=-]{40,}$")

# Placeholders allowed per section. The runner must provide every one it is given.
TRAIN_PLACEHOLDERS = {
    "task", "num_envs", "max_iters", "iters", "save_every", "run_id", "run_name",
    "wandb_entity", "wandb_project", "repo_dir", "video_flags", "resume_args",
    "video_every", "video_interval", "video_length",
}
INSTALL_PLACEHOLDERS = {"repo_dir", "task", "run_id", "run_name", "wandb_entity", "wandb_project"}
BENCH_PLACEHOLDERS = TRAIN_PLACEHOLDERS - {"video_flags", "resume_args"} | {"checkpoint", "bench_json"}
VIDEO_PLACEHOLDERS = {"video_every", "video_interval", "video_length", "save_every"}
RESUME_ARG_PLACEHOLDERS = {"resume_run", "resume_checkpoint", "resume_path", "resume_iter"}
RESTORE_PLACEHOLDERS = {"run_name", "experiment", "run_id"}
GPU_TYPES = ("T4", "L4", "A100", "H100")
ACCELERATORS = ("GPU", "none")
# Sections whose keys are free-form names, not fixed keys.
FREE_MAPS = {"env", "secret_env"}
EXAMPLE_TRAIN = "python train.py --steps {iters} --run-name {run_id}{resume_args}"

DEFAULTS: dict[str, Any] = {
    "description": "",
    "repo": {"url": None, "branch": "main", "token_secret": None, "token_user": "x-access-token",
             "dir": None, "submodules": False, "commit": None,
             "fallback_url": None, "fallback_branch": None, "fallback_token_secret": None,
             "fallback_token_user": "x-access-token"},
    "python": None,
    "install": [],
    "mujoco_gl": False,
    "env": {},
    "secret_env": {},
    # wandb.mode auto: online when wandb.entity is set, else disabled (no W&B account needed).
    "wandb": {"mode": "auto", "entity": None, "project": None,
              "api_key_secret": "WANDB_API_KEY", "iteration_key": "_step"},
    "run": {"prefix": None, "name": "1"},
    "train": {"command": None, "task": "", "num_envs": 1024, "max_iters": 3000, "save_every": 100},
    # Parse training stdout without W&B. regex: one group (or named group "iter") = the iteration;
    # optional named group "total". Matching lines are logged as "[colab-train] progress: ...".
    "progress": {"regex": None},
    "video": {"every": 0, "steps_per_iter": 1, "length": 400, "flags": ""},
    # resume.source auto: none without checkpoint_glob; else wandb when W&B is online, else drive.
    # iters_mode total: train.max_iters is passed unchanged on resume. remaining: max_iters minus the loaded
    # iteration (for loops that restart at the loaded iteration, e.g. rsl_rl). experiment_config_key: the W&B
    # config key whose parent folder names {experiment} (null: the W&B project name). wandb_config_shim: let
    # the config update that a resumed W&B run makes through (rsl_rl writes one; see runner.py).
    "resume": {"source": "auto", "checkpoint_glob": None,
               "checkpoint_regex": r"model_(\d+)\.pt", "restore_dir": "checkpoints/{run_name}",
               "experiment_config_key": None, "wandb_config_shim": False,
               "sync_extra": [], "iters_mode": "total", "args": ""},
    "bench": {"command": None},
    "outputs": [],
    "limits": {"max_hours": 11.0},
    "runtime": {"accelerator": "GPU", "gpu_type": "T4"},
}


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def repo_dir_name(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(text or ""))


def _check_placeholders(where: str, text: str, allowed: set[str], errors: list[str]) -> None:
    unknown = placeholders(text) - allowed
    if unknown:
        errors.append(f"{where}: unknown placeholder(s) {sorted(unknown)}; allowed {sorted(allowed)}")


def _unknown_keys(raw: dict, base: dict, path: str, errors: list[str]) -> None:
    """Flag keys that no section defines, with the closest valid name as a hint."""
    for k, v in raw.items():
        where = f"{path}{k}"
        if k not in base:
            close = difflib.get_close_matches(str(k), list(base), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            errors.append(f"{where}: unknown key{hint}. Allowed here: {', '.join(sorted(base))}")
            continue
        if isinstance(base[k], dict) and k not in FREE_MAPS and isinstance(v, dict):
            _unknown_keys(v, base[k], f"{where}.", errors)


def _looks_like_token(text: str) -> bool:
    return bool(SECRET_LIKE_VALUE_RE.match(text) or PLAIN_LONG_TOKEN_RE.match(text))


def validate(raw: Any, run_name: str | None = None) -> tuple[dict, list[str]]:
    """Return (normalized job, errors). Normalized job has a `derived` block with run ids and paths.

    Error messages name the exact key and, where it helps, the command to fix it. Values of secrets never
    appear in them: a secret is referenced by its NAME only."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return {}, ["job spec must be a YAML mapping (key: value lines), see `colab-train init`"]
    _unknown_keys(raw, {**DEFAULTS, "name": None}, "", errors)
    # An empty section (`wandb:` with nothing under it) means "defaults".
    raw = {k: ({} if (v is None and isinstance(DEFAULTS.get(k), dict)) else v) for k, v in raw.items()}
    job = _merge(DEFAULTS, {k: v for k, v in raw.items() if k != "name"})
    job["name"] = raw.get("name")
    name = job.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,48}", name or ""):
        errors.append("name: required, lowercase letters, digits and dashes (2-49 chars), e.g. name: my-ppo-run")
    repo = job["repo"]
    url = repo.get("url")
    if not isinstance(url, str) or not re.match(r"^https://[^\s]+$", url or ""):
        errors.append("repo.url: required https URL of the git repo, e.g. https://github.com/<you>/<repo>.git "
                      "(SSH git@... URLs and credentials inside the URL are not supported)")
    elif re.search(r"://[^/@]+:[^/@]+@", url):
        errors.append("repo.url contains credentials; use repo.token_secret instead")
    ts = repo.get("token_secret")
    if ts and (not SECRET_NAME_RE.match(str(ts)) or _looks_like_token(str(ts))):
        errors.append("repo.token_secret: the NAME of a Colab secret (e.g. GITHUB_TOKEN), never the token itself. "
                      "Store the value with `colab-train secrets set GITHUB_TOKEN`")
    if repo.get("token_user") and not re.fullmatch(r"[A-Za-z0-9._@-]+", str(repo["token_user"])):
        errors.append("repo.token_user: letters, digits, . _ @ - only (GitHub: x-access-token; other hosts: see their docs)")
    outs = raw.get("outputs") if isinstance(raw, dict) else None
    if outs is not None and (not isinstance(outs, list) or not all(isinstance(o, str) and o.strip() for o in outs)):
        errors.append("outputs: a list of glob patterns relative to the repo, e.g. [\"bench/**/*.json\"]")
    elif outs and any(o.startswith("/") or ".." in Path(o).parts for o in outs):
        errors.append("outputs: patterns are relative to the repo (no leading / and no ..)")
    if repo.get("commit") is not None and not re.fullmatch(r"[0-9a-f]{7,40}", str(repo["commit"])):
        errors.append("repo.commit: a git commit hash (7 to 40 hex digits) to pin the job to, e.g. 6486b99")
    if repo.get("dir") is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", str(repo["dir"])):
        errors.append("repo.dir: a single directory name, e.g. my-repo")
    # Optional second host, used only when the primary clone or fetch fails. Same repo, other URL.
    fb = repo.get("fallback_url")
    if fb is not None:
        if not isinstance(fb, str) or not re.match(r"^https://[^\s]+$", fb):
            errors.append("repo.fallback_url: https URL (e.g. https://github.com/<user>/<repo>.git)")
        elif re.search(r"://[^/@]+:[^/@]+@", fb):
            errors.append("repo.fallback_url contains credentials; use repo.fallback_token_secret instead")
    if repo.get("fallback_token_user") and not re.fullmatch(r"[A-Za-z0-9._@-]+", str(repo["fallback_token_user"])):
        errors.append("repo.fallback_token_user: letters, digits, . _ @ - only")
    if repo.get("fallback_token_secret") and not SECRET_NAME_RE.match(str(repo["fallback_token_secret"])):
        errors.append("repo.fallback_token_secret: the NAME of a Colab secret, never the token itself")

    install = job["install"]
    if not isinstance(install, list) or not all(isinstance(c, str) for c in install):
        errors.append("install: a list of shell commands, e.g. install: [\"pip install -r requirements.txt\"]")
        install = []
    wb = job["wandb"]
    if wb.get("mode") == "auto":
        wb["mode"] = "online" if wb.get("entity") else "disabled"
    if wb["mode"] not in ("online", "offline", "disabled"):
        errors.append("wandb.mode: online | offline | disabled (or auto: online if wandb.entity is set)")
    if wb["mode"] == "online" and not wb.get("entity"):
        errors.append("wandb.entity: your W&B user or team name (wandb.ai, top right). No W&B account? "
                      "set wandb.mode: disabled")
    if wb.get("api_key_secret") and not SECRET_NAME_RE.match(str(wb["api_key_secret"])):
        errors.append("wandb.api_key_secret: the NAME of a Colab secret (default WANDB_API_KEY)")

    train = job["train"]
    if not train.get("command"):
        errors.append(f"train.command: required, the command that starts training. Example: "
                      f"train.command: \"{EXAMPLE_TRAIN}\" (placeholders such as {{iters}} are filled in by the runner)")
    elif not isinstance(train.get("command"), str):
        errors.append("train.command: a single string, e.g. \"python train.py --steps {iters}\"")
    for key in ("num_envs", "max_iters", "save_every"):
        if not isinstance(train.get(key), int) or train[key] < 0:
            errors.append(f"train.{key}: non-negative integer")
    if not isinstance(train.get("max_iters"), int) or train["max_iters"] < 1:
        errors.append("train.max_iters: must be at least 1 (the total number of training iterations or steps)")
    if not isinstance(train.get("save_every"), int) or train["save_every"] < 1:
        errors.append("train.save_every: must be at least 1 (checkpoints are the resume point)")

    prog = job["progress"].get("regex")
    if prog:
        try:
            rx = re.compile(prog)
            if rx.groups < 1 and "iter" not in rx.groupindex:
                errors.append("progress.regex: needs a group for the iteration, e.g. 'step (\\d+)/\\d+' "
                              "or a named group (?P<iter>\\d+)")
        except re.error as e:
            errors.append(f"progress.regex: invalid regex ({e})")

    video = job["video"]
    if not isinstance(video.get("every"), int) or video["every"] < 0:
        errors.append("video.every: 0 disables videos, otherwise iterations between videos")
    elif video["every"] > 0 and not video.get("flags"):
        errors.append("video.flags: required when video.every > 0 (flags added to the train command)")

    resume = job["resume"]
    if resume["source"] not in ("auto", "wandb", "drive", "none"):
        errors.append("resume.source: auto | wandb | drive | none")
    if resume["source"] == "auto":
        if not resume.get("checkpoint_glob"):
            resume["source"] = "none"  # nothing says which files to resume from
        elif wb["mode"] == "online":
            resume["source"] = "wandb"
        else:
            resume["source"] = "drive"
    if resume["source"] != "none":
        if not resume.get("checkpoint_glob"):
            errors.append("resume.checkpoint_glob: required when resume.source is not none, e.g. "
                          "\"checkpoints/**/ckpt_*.pt\" (the files your trainer writes)")
        if resume["source"] == "wandb" and wb["mode"] != "online":
            errors.append("resume.source: wandb needs wandb.mode online (wandb.entity set). Without W&B use "
                          "resume.source: drive (checkpoints on Google Drive) or none")
    try:
        if re.compile(resume["checkpoint_regex"]).groups < 1:
            errors.append("resume.checkpoint_regex: needs one group for the iteration, e.g. model_(\\d+)\\.pt")
    except re.error as e:
        errors.append(f"resume.checkpoint_regex: invalid regex ({e})")
    if resume["iters_mode"] not in ("remaining", "total"):
        errors.append("resume.iters_mode: remaining | total")
    ekey = resume.get("experiment_config_key")
    if ekey is not None and not (isinstance(ekey, str) and SECRET_NAME_RE.match(ekey)):
        errors.append("resume.experiment_config_key: a W&B config key name (e.g. log_dir) or null")
    if not isinstance(resume.get("wandb_config_shim"), bool):
        errors.append("resume.wandb_config_shim: true or false")

    for k, v in job["env"].items():
        if not SECRET_NAME_RE.match(str(k)):
            errors.append(f"env: bad variable name {k!r}")
        elif SECRET_LIKE_KEY_RE.search(str(k)) or SECRET_LIKE_VALUE_RE.match(str(v)) \
                or PLAIN_LONG_TOKEN_RE.match(str(v)):
            errors.append(f"env.{k}: looks like a secret; put the Colab secret NAME in secret_env instead, "
                          f"e.g. secret_env: {{{k}: MY_SECRET}}")
    for k, v in job["secret_env"].items():
        if not SECRET_NAME_RE.match(str(k)) or not SECRET_NAME_RE.match(str(v)):
            errors.append(f"secret_env: {k!r}: {v!r} must be two NAMES: the variable and the Colab secret")

    if job["limits"]["max_hours"] is not None and not (0 < float(job["limits"]["max_hours"]) <= 24):
        errors.append("limits.max_hours: between 0 and 24 (a free Colab runtime lasts at most ~12 h)")

    rt = job["runtime"]
    if rt.get("accelerator") not in ACCELERATORS:
        errors.append("runtime.accelerator: GPU (the usual choice) or none (CPU only)")
    elif rt["accelerator"] == "GPU" and rt.get("gpu_type") not in GPU_TYPES:
        errors.append(f"runtime.gpu_type: one of {', '.join(GPU_TYPES)} (free Colab mostly assigns T4)")

    for i, cmd in enumerate(install):
        _check_placeholders(f"install[{i}]", cmd, INSTALL_PLACEHOLDERS, errors)
    _check_placeholders("train.command", train.get("command") or "", TRAIN_PLACEHOLDERS, errors)
    _check_placeholders("video.flags", video.get("flags") or "", VIDEO_PLACEHOLDERS, errors)
    _check_placeholders("resume.args", resume.get("args") or "", RESUME_ARG_PLACEHOLDERS, errors)
    _check_placeholders("resume.restore_dir", resume.get("restore_dir") or "", RESTORE_PLACEHOLDERS, errors)
    if job["bench"].get("command"):
        _check_placeholders("bench.command", job["bench"]["command"], BENCH_PLACEHOLDERS, errors)

    if run_name is not None:
        job["run"]["name"] = str(run_name)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(job["run"]["name"])):
        errors.append("run.name: letters, digits, . _ - only (it becomes a W&B run name)")

    if errors:
        return job, errors
    dir_name = repo.get("dir") or repo_dir_name(url)
    prefix = job["run"]["prefix"] or name
    run_id = f"{prefix}-{job['run']['name']}"
    job["derived"] = {
        "run_id": run_id,
        "repo_dir_name": dir_name,
        "wandb_project": wb.get("project") or dir_name.lower(),
        "wandb_entity": wb.get("entity") or "",
    }
    return job, []


def load(path: str | Path, run_name: str | None = None) -> dict:
    """Load and validate a YAML job spec file."""
    return load_text(Path(path).read_text(), run_name=run_name)


def load_text(text: str, run_name: str | None = None) -> dict:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SpecError([f"YAML parse error: {e}"]) from e
    job, errors = validate(raw, run_name=run_name)
    if errors:
        raise SpecError(errors)
    return job


def preset_names() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def preset_text(name: str) -> str:
    path = PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        raise SpecError([f"no preset {name!r}; presets: {', '.join(preset_names())}"])
    return path.read_text()


def resolve_job_arg(arg: str) -> Path:
    """A job argument is a YAML path, a jobs/<name>.yaml in the cwd, or a built-in preset name."""
    p = Path(arg)
    if p.is_file():
        return p
    local = Path.cwd() / "jobs" / f"{arg}.yaml"
    if local.is_file():
        return local
    if arg in preset_names():
        return PRESET_DIR / f"{arg}.yaml"
    if p.suffix in (".yaml", ".yml"):
        raise SpecError([f"job file {arg!r} not found (check the path; run from the repo root)"])
    raise SpecError([f"job {arg!r}: not a file, not jobs/{arg}.yaml, not a preset ({', '.join(preset_names())})"])
