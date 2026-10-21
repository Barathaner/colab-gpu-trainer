"""Local store for Colab Secret VALUES, used only to fill the Colab Secrets panel.

Values live in ~/.config/colab-train/secrets.json (directory 0700, file 0600), or in the OS keyring when the optional
`keyring` package has a real backend (COLAB_TRAIN_SECRET_BACKEND=file|keyring|auto, default auto). Nothing in this
module prints a value. `list` shows names, source, backend and time only. Values leave the module only through
get(), and only the browser's Secrets panel calls it.
"""
from __future__ import annotations

import datetime as dt
import json
import netrc
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SERVICE = "colab-train"
SECRET_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
WANDB_HOST = "api.wandb.ai"


class SecretsError(RuntimeError):
    """Safe to print: it never contains a value."""


def config_dir() -> Path:
    override = os.environ.get("COLAB_TRAIN_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".config" / "colab-train"


def store_path() -> Path:
    return config_dir() / "secrets.json"


def check_name(name: str) -> str:
    if not SECRET_NAME_RE.match(name or ""):
        raise SecretsError(f"{name!r}: a Colab Secret name is letters, digits and underscore (not starting with a digit)")
    return name


def _keyring() -> Any:
    """The keyring module when a real backend exists, else None. Forced off with COLAB_TRAIN_SECRET_BACKEND=file."""
    if os.environ.get("COLAB_TRAIN_SECRET_BACKEND", "auto") == "file":
        return None
    try:
        import keyring  # type: ignore
    except ImportError:
        return None
    try:
        kr = keyring.get_keyring()
    except Exception:  # noqa: BLE001 - a broken backend means: not available
        return None
    return keyring if getattr(kr, "priority", 0) > 0 else None


def _read(path: Path | None = None) -> dict:
    p = path or store_path()
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"secrets": {}}
    return data if isinstance(data, dict) and isinstance(data.get("secrets"), dict) else {"secrets": {}}


def _write(data: dict, path: Path | None = None) -> None:
    p = path or store_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(p.parent, 0o700)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(p)


def set_value(name: str, value: str, source: str = "set", path: Path | None = None) -> str:
    """Store a value. Returns the backend used ('keyring' or 'file'). The value is never returned or printed."""
    check_name(name)
    if not value or "\n" in value or "\r" in value:
        raise SecretsError(f"{name}: the value is empty or has a line break")
    data = _read(path)
    kr = _keyring()
    backend = "keyring" if kr else "file"
    entry = {"source": source, "backend": backend,
             "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if kr:
        kr.set_password(SERVICE, name, value)
    else:
        entry["value"] = value
    data["secrets"][name] = entry
    _write(data, path)
    return backend


def get(name: str, path: Path | None = None) -> str | None:
    """The stored value, for the Secrets panel only. Never print the result."""
    entry = _read(path)["secrets"].get(check_name(name))
    if not entry:
        return None
    if entry.get("backend") == "keyring":
        kr = _keyring()
        return kr.get_password(SERVICE, name) if kr else None
    value = entry.get("value")
    return value if isinstance(value, str) and value else None


def names(path: Path | None = None) -> dict[str, dict]:
    """name -> {source, backend, updated}. No values."""
    out = {}
    for name, e in sorted(_read(path)["secrets"].items()):
        out[name] = {"source": e.get("source", "?"), "backend": e.get("backend", "file"),
                     "updated": e.get("updated", "?")}
    return out


def rm(name: str, path: Path | None = None) -> bool:
    data = _read(path)
    entry = data["secrets"].pop(check_name(name), None)
    if entry is None:
        return False
    if entry.get("backend") == "keyring" and _keyring():
        try:
            _keyring().delete_password(SERVICE, name)
        except Exception:  # noqa: BLE001 - the index entry is gone either way
            pass
    _write(data, path)
    return True


def missing(needed: list[str], path: Path | None = None) -> list[str]:
    have = set(_read(path)["secrets"])
    return [n for n in needed if n not in have]


# ---------------------------------------------------------------- import (values never shown)

def import_wandb(netrc_path: Path | None = None, env: dict | None = None, path: Path | None = None) -> str:
    """WANDB_API_KEY from the WANDB_API_KEY env var, else the api.wandb.ai entry of ~/.netrc."""
    env = os.environ if env is None else env
    if env.get("WANDB_API_KEY"):
        set_value("WANDB_API_KEY", env["WANDB_API_KEY"], source="import:env", path=path)
        return "env WANDB_API_KEY"
    nrc = netrc_path or Path.home() / ".netrc"
    try:
        auth = netrc.netrc(str(nrc)).authenticators(WANDB_HOST)
    except (OSError, netrc.NetrcParseError) as e:
        raise SecretsError(f"no W&B key: {nrc} is missing or unreadable ({type(e).__name__})") from e
    if not auth or not auth[2]:
        raise SecretsError(f"no W&B key: {nrc} has no entry for {WANDB_HOST}")
    set_value("WANDB_API_KEY", auth[2], source="import:netrc", path=path)
    return f"~/.netrc entry for {WANDB_HOST}"


def import_git_credential(host: str, name: str, path: Path | None = None) -> str:
    """Ask git's credential helpers for the password of https://host. Never prompts (GIT_TERMINAL_PROMPT=0)."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        res = subprocess.run(["git", "credential", "fill"], input=f"protocol=https\nhost={host}\n\n",
                             capture_output=True, text=True, env=env, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SecretsError(f"git credential fill failed ({type(e).__name__})") from e
    password = None
    for line in res.stdout.splitlines():
        if line.startswith("password="):
            password = line[len("password="):]
    if res.returncode != 0 or not password:
        raise SecretsError(f"git has no stored password for https://{host} (set one with a git credential helper, "
                           "or use --from-file)")
    set_value(check_name(name), password, source=f"import:git-credential:{host}", path=path)
    return f"git credential helper for {host}"


def import_file(file: Path, name: str, path: Path | None = None) -> str:
    try:
        value = file.read_text().strip()
    except OSError as e:
        raise SecretsError(f"cannot read {file} ({type(e).__name__})") from e
    set_value(check_name(name), value, source=f"import:file", path=path)
    return f"file {file.name}"


# ---------------------------------------------------------------- scrub

def scrub(text: str, values: list[str | None]) -> str:
    """Replace any known value in text (an error message, a log line) before it is shown."""
    for v in values:
        if v:
            text = text.replace(v, "***")
    return text
