"""Fill Colab's Secrets panel from the local store (see secrets.py). Runs inside an open notebook tab.

Rules, all enforced here:
- The value goes from the store straight into the page with fill(). It is never read back from the page, never
  printed, and never put in an exception message (errors are scrubbed).
- No screenshot is taken while the panel is open. Failures raise PanelError, which the caller reports as text.
- Existing secrets are checked by their name only (the "Delete secret NAME" button and the notebook-access switch).
  Their values are not touched unless update=True.
- Only the named secrets are touched. Other secrets in the account (for example "WANDB_API_KEY") are left alone.

Labels are English (the notebook is opened with ?hl=en); the German alternatives are also accepted.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from . import secrets
from .browser import BrowserError

PANEL_BUTTON = re.compile(r"^(Secrets|Geheimnisse)$")
ADD_TEXT = re.compile(r"Add new secret|Neues Geheimnis hinzuf", re.I)
NEW_NAME_LABEL = re.compile(r"^(Name of new secret|Name des neuen Geheimnisses)$")
NEW_VALUE_LABEL = re.compile(r"^(Value of new secret|Wert des neuen Geheimnisses)$")


class PanelError(BrowserError):
    """A panel step failed. The message never contains a value. No screenshot is taken for it."""


def _scrub_exc(e: Exception, value: str | None) -> str:
    return secrets.scrub(f"{type(e).__name__}: {e}", [value])


def _panel_open(page: Any) -> bool:
    h = page.get_by_role("heading", name=PANEL_BUTTON).first
    return h.count() > 0 and h.is_visible()


def open_panel(page: Any) -> None:
    try:
        if _panel_open(page):
            return
        page.get_by_role("button", name=PANEL_BUTTON).first.click()
        page.get_by_role("heading", name=PANEL_BUTTON).first.wait_for(state="visible", timeout=10000)
    except Exception as e:  # noqa: BLE001
        raise PanelError(f"could not open the Secrets panel: {_scrub_exc(e, None)}") from None


def close_panel(page: Any) -> None:
    """Escape first; if the panel is still open, the Secrets button toggles it shut. Best effort, no screenshot."""
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
        if _panel_open(page):
            page.get_by_role("button", name=PANEL_BUTTON).first.click()
            page.wait_for_timeout(500)
    except Exception:  # noqa: BLE001 - closing is best effort
        pass


def _row_button(page: Any, name: str, verb: str) -> Any:
    return page.get_by_role("button", name=re.compile(rf'^{verb} secret "{re.escape(name)}"'))


def exists(page: Any, name: str) -> bool:
    return _row_button(page, name, "Delete").count() > 0


def _access_box(page: Any, name: str) -> Any:
    # The notebook-access control is a checkbox (inside a switch component); it is found by its label.
    return page.get_by_label(re.compile(rf'^Toggle notebook access for secret "{re.escape(name)}"'))


def access_on(page: Any, name: str) -> bool:
    box = _access_box(page, name)
    return box.first.is_checked() if box.count() else False


def _ensure_access(page: Any, name: str) -> bool:
    if not access_on(page, name):
        _access_box(page, name).first.click()
        page.wait_for_timeout(800)
    return access_on(page, name)


def _add(page: Any, name: str, value: str) -> None:
    """Add new secret: the name field, then the value textarea. Colab saves the row by itself ("All changes saved")."""
    page.get_by_text(ADD_TEXT).first.click()
    name_box = page.get_by_label(NEW_NAME_LABEL)
    name_box.first.wait_for(state="visible", timeout=10000)
    name_box.first.fill(name)
    value_box = page.get_by_label(NEW_VALUE_LABEL)
    value_box.first.wait_for(state="visible", timeout=10000)
    value_box.first.fill(value)
    deadline = time.monotonic() + 60  # Colab's autosave can take several seconds
    while time.monotonic() < deadline and not exists(page, name):
        page.wait_for_timeout(1000)
    if not exists(page, name):
        raise PanelError(f"secret {name}: Colab did not confirm the save within 60 s. Run the same command again: "
                         "a saved secret is found by name and is not added twice")


def _update(page: Any, name: str, value: str) -> None:
    box = page.get_by_label(re.compile(rf'^Value of secret "{re.escape(name)}"'))
    box.first.fill(value)
    page.wait_for_timeout(800)
    save = page.get_by_role("button", name=re.compile(r"^(Save|Speichern)$"))
    if save.count():
        save.first.click()


def ensure_secrets(page: Any, needed: list[str], update: bool = False,
                   log: Callable[[str], None] = print) -> dict[str, str]:
    """Make sure each needed secret exists in the panel with notebook access ON. Returns name -> what was done.

    Values come from the local store. A needed name missing from the store is an error (no value to fill)."""
    gone = secrets.missing(needed)
    if gone:
        raise PanelError("no local value for " + ", ".join(gone) + ": set them with `colab-train secrets set NAME`")
    done: dict[str, str] = {}
    open_panel(page)
    try:
        for name in needed:
            value = secrets.get(name)
            if not value:
                raise PanelError(f"secret {name}: the local store returned no value")
            try:
                if not exists(page, name):
                    _add(page, name, value)
                    done[name] = "added"
                elif update:
                    _update(page, name, value)
                    done[name] = "updated"
                else:
                    done[name] = "already in Colab"
                if not _ensure_access(page, name):
                    raise PanelError(f"secret {name}: notebook access did not turn on")
                done[name] += ", notebook access on"
                log(f"secret {name}: {done[name]}")
            except PanelError:
                raise
            except Exception as e:  # noqa: BLE001 - scrub: the value must not leak through an error message
                raise PanelError(f"secret {name}: {_scrub_exc(e, value)}") from None
            finally:
                value = None
    finally:
        close_panel(page)
    return done
