"""Put a rendered notebook where Colab can open it.

auto  : secret GitHub gist via the `gh` CLI when it is logged in, else a local file (default).
gist  : secret gist, error if gh is not usable.
local : local file only; the user uploads it in Colab (File > Upload notebook).
A secret gist is unlisted, not private: it is reachable by anyone with its URL. The notebook holds
Colab Secret names only, never values, but do not put job specs with private details in it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class PublishError(RuntimeError):
    pass


def gh_ready() -> bool:
    if shutil.which("gh") is None:
        return False
    return subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0


def gist_colab_url(path: Path, gist_url: str, login: str) -> str:
    m = re.search(r"gist\.github\.com/[^/\s]+/([0-9a-f]+)", gist_url)
    if not m:
        raise PublishError(f"unexpected gh output: {gist_url!r}")
    return f"https://colab.research.google.com/gist/{login}/{m.group(1)}/{path.name}"


def publish(path: Path, mode: str, description: str) -> dict:
    """Return {"mode": "gist"|"local", "colab_url": str|None, "gist_url": str|None}."""
    if mode == "local":
        return {"mode": "local", "colab_url": None, "gist_url": None}
    if not gh_ready():
        if mode == "gist":
            raise PublishError("gist requested but gh is missing or not logged in (gh auth login)")
        return {"mode": "local", "colab_url": None, "gist_url": None}
    try:
        out = subprocess.run(["gh", "gist", "create", "--desc", description, str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        login = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as e:
        if mode == "gist":
            raise PublishError(f"gh gist create failed (exit {e.returncode})") from e
        return {"mode": "local", "colab_url": None, "gist_url": None}
    return {"mode": "gist", "gist_url": out, "colab_url": gist_colab_url(path, out, login)}
