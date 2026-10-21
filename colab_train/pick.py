"""Which job `launch --browser` trains when no JOB is named, and a run name nobody has used yet.

Pure functions: the CLI passes in the job names that validate, the priority list from config, the user's answer
and the run ids already seen (state file plus W&B). Nothing here touches the network.
"""
from __future__ import annotations

import re
from typing import Callable


class PickError(ValueError):
    pass


def choose_auto(available: list[str], priority: list[str]) -> tuple[str, str]:
    """First job of the priority list that is available, else the first available job. Returns (job, why)."""
    if not available:
        raise PickError("no valid job found (jobs/*.yaml or a preset)")
    for name in priority:
        if name in available:
            return name, f"first available in the priority list ({', '.join(priority)})"
    return available[0], "none of the priority list is available; first available job"


def choose_interactive(available: list[str], ask: Callable[[str], str] = input,
                       out: Callable[[str], None] = print) -> str:
    """Numbered menu. Empty answer or a bad number is an error, so nothing starts by accident."""
    if not available:
        raise PickError("no valid job found (jobs/*.yaml or a preset)")
    out("which job to train on Colab?")
    for i, name in enumerate(available, 1):
        out(f"  {i}. {name}")
    answer = ask("number: ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(available):
        raise PickError(f"not a job number: {answer!r}")
    return available[int(answer) - 1]


def fresh_run_name(prefix: str, base: str, used_run_ids: set[str]) -> str:
    """`base` if prefix-base is unused, else base with the number raised until the run id is new."""
    if not _taken(f"{prefix}-{base}", used_run_ids):
        return base
    m = re.fullmatch(r"(.*?)-(\d+)", base)
    stem, n = (m.group(1), int(m.group(2))) if m else (base, 1)
    while True:
        n += 1
        cand = f"{stem}-{n}"
        if not _taken(f"{prefix}-{cand}", used_run_ids):
            return cand


def _taken(run_id: str, used: set[str]) -> bool:
    """W&B display names are often timestamp-prefixed, so a run name that ends with the run id counts as taken."""
    return any(u == run_id or u.endswith(run_id) for u in used)
