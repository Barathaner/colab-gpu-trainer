"""Colab web steps for `launch --browser` and `stop --browser`, driven over CDP (see browser.py).

The current Colab UI hides the File/Runtime menubar by default, so the steps use the visible controls:
the home "Upload notebook" button, the toolbar "Run all" button, and the "Additional connection options" menu
(the arrow next to Connect) with "Change runtime type" and "Disconnect and delete runtime".

Language: every URL carries ?hl=en, so the labels are English. The regexes also accept the German labels, in case
Colab ignores the parameter. Selectors use roles and accessible names, never the menubar.

Stop conditions (captcha, robot check, usage limit, Google sign-in) raise browser.SafetyStop and are never clicked
through. The only dialog buttons clicked are the approved ones: "Run anyway" and "Grant access" (Secrets). Connect is
never clicked, because it starts a runtime. Run all is the single action that starts one.

Verified on the live logged-in Colab (read-only probes) and on a local fake page (tests/test_browser_ui.py).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from . import browser
from .browser import BrowserError, SafetyStop

NOTEBOOK_URL = re.compile(r"/(notebooks|drive)/")  # Colab notebook pages: /drive/<id> or /notebooks/<id>
START_TIMEOUT_S = 900  # Colab can take minutes to assign a T4
STUCK_S = 300  # no runner line this long after Run all counts as stuck (one retry, see cli._drive_launch)


class RunnerStuck(browser.BrowserError):
    """Run all was accepted but the runner printed nothing in time. The caller may retry once."""

UPLOAD_NAMES = re.compile(r"^(Upload notebook|Notebook hochladen)$")
RUN_ALL_NAMES = re.compile(r"^(Run all|Alle ausführen)$")
OPTIONS_NAMES = re.compile(r"^(Additional connection options|Zusätzliche Verbindungsoptionen)$")
RUNTIME_TYPE_NAMES = re.compile(r"^(Change runtime type|Laufzeittyp ändern)")
DISCONNECT_NAMES = re.compile(r"^(Disconnect and delete runtime|Laufzeit trennen und löschen)")
T4_RADIO_NAMES = re.compile(r"^T4 GPU")
SAVE_NAMES = re.compile(r"^(Save|Speichern)$")
CANCEL_NAMES = re.compile(r"^(Cancel|Abbrechen)$")
RUN_ANYWAY_NAMES = ("Run anyway", "Trotzdem ausführen")
GRANT_ACCESS_NAMES = ("Grant access", "Zugriff gewähren")


def open_home(ctx: Any) -> Any:
    """A new tab on the Colab home page, English. Ready when the Upload button or a Sign in link shows."""
    page = ctx.new_page()
    page.set_default_timeout(20000)
    page.goto(browser.COLAB_URL, wait_until="domcontentloaded")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if page.get_by_role("button", name=UPLOAD_NAMES).count() or \
                page.get_by_role("link", name=re.compile(r"^(Sign in|Anmelden)$")).count():
            break
        page.wait_for_timeout(1000)
    return page


def _visible_button(page: Any, name: Any) -> Any:
    loc = page.get_by_role("button", name=name)
    for i in range(loc.count()):
        if loc.nth(i).is_visible():
            return loc.nth(i)
    return None


def click_dialog_button(page: Any, names: tuple[str, ...]) -> bool:
    """Click a visible button by name if present. Only used for the two approved prompts."""
    for name in names:
        btn = _visible_button(page, re.compile(rf"^{re.escape(name)}$"))
        if btn is not None:
            btn.click()
            return True
    return False


def upload_notebook(page: Any, nb_path: Path) -> Any:
    """Upload notebook on the home page, set the file, return the page that holds the notebook.

    Colab opens the notebook in a new tab (or navigates this one). Tabs are compared by URL, because Playwright
    hands out new Python objects for the same tab."""
    ctx = page.context
    start_url = page.url or ""
    before_urls = [p.url for p in ctx.pages]
    btn = _visible_button(page, UPLOAD_NAMES)
    if btn is None:
        raise BrowserError("no 'Upload notebook' button on the Colab home page (see the screenshot)")
    btn.click()
    inp = page.locator('input[type="file"]').first
    inp.wait_for(state="attached", timeout=10000)
    inp.set_input_files(str(nb_path))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if NOTEBOOK_URL.search(page.url or "") and not NOTEBOOK_URL.search(start_url):
            page.bring_to_front()
            page.wait_for_load_state("domcontentloaded")
            return page
        for p in ctx.pages:
            if NOTEBOOK_URL.search(p.url or "") and p.url not in before_urls:
                p.bring_to_front()
                p.wait_for_load_state("domcontentloaded")
                return p
        page.wait_for_timeout(1000)  # pumps Playwright events; time.sleep would freeze the URL updates
    raise BrowserError(f"the notebook did not open after the upload; tabs: {[p.url for p in ctx.pages]}")


def _open_options(page: Any) -> None:
    btn = page.get_by_role("button", name=OPTIONS_NAMES).first
    btn.wait_for(state="visible", timeout=20000)
    btn.click()


def ensure_t4(page: Any, gpu_type: str = "T4") -> str:
    """Change runtime type: make sure the job's GPU (runtime.gpu_type, default T4) is the chosen accelerator.
    Save only when it changed, else Cancel. Free Colab may not offer every type; then this raises.

    Returns the chosen accelerator label. Never clicks Connect, which would start a runtime."""
    radio_names = re.compile(rf"^{re.escape(gpu_type)} GPU")
    _open_options(page)
    page.get_by_role("menuitem", name=RUNTIME_TYPE_NAMES).first.click()
    t4 = page.get_by_role("radio", name=radio_names).first
    t4.wait_for(state="visible", timeout=10000)
    if not t4.is_checked():
        t4.click()
        page.get_by_role("button", name=SAVE_NAMES).last.click()
    else:
        page.get_by_role("button", name=CANCEL_NAMES).last.click()
    page.wait_for_timeout(1000)
    if page.get_by_role("radio", name=radio_names).count() and \
            page.get_by_role("radio", name=radio_names).first.is_visible():
        raise BrowserError("the runtime type dialog did not close after Save/Cancel (see the screenshot)")
    return f"{gpu_type} GPU"


def page_text(page: Any) -> str:
    """Visible text of the page AND its frames. Colab renders cell output in cross-origin outputframe iframes, which
    the main document's innerText does not contain."""
    parts = []
    for f in page.frames:
        try:
            parts.append(f.evaluate("() => document.body ? document.body.innerText : ''"))
        except Exception:  # a frame can detach while we read it
            continue
    return "\n".join(parts)


def output_lines(page: Any) -> list[str]:
    # Line start only: the notebook's own code has "[colab-train]" inside print(...), which is not at line start.
    return [ln.rstrip() for ln in page_text(page).splitlines() if ln.startswith("[colab-train]")]


def run_all(page: Any, log: Callable[[str], None] = print, stuck_s: int | None = STUCK_S) -> list[str]:
    """Run all (toolbar button), then wait for the runner's first lines. Clicks only the approved dialogs."""
    btn = page.get_by_role("button", name=RUN_ALL_NAMES).first
    if btn.count():
        btn.click()
    else:
        page.keyboard.press("Control+F9")  # the Run all shortcut, when the toolbar button is not there
    return wait_for_runner(page, log, stuck_s=stuck_s)


class SessionLimit(browser.SafetyStop):
    """Colab refused a second runtime: this account runs one session at a time. Not a failure: re-queue."""


SESSION_LIMIT = re.compile(r"too many (active )?sessions", re.I)


def session_limit(page: Any) -> bool:
    """Only VISIBLE dialogs count: hidden text on the page would match forever."""
    return any(SESSION_LIMIT.search(t) for t in browser.dialog_texts(page))


def dismiss_session_limit(page: Any) -> bool:
    """Click the dialog's Cancel. Never 'Manage sessions': that would terminate the other runtime."""
    btn = _visible_button(page, re.compile(r"^(Cancel|Abbrechen)$"))
    if btn is None:
        return False
    btn.click()
    page.wait_for_timeout(800)
    return True


_NO_GPU = re.compile(r"No such file or directory: 'nvidia-smi'|no GPU attached to this runtime")


def no_gpu(page: Any) -> bool:
    """True when the runtime has no GPU: the cell's nvidia-smi call failed, or the runner said so."""
    return bool(_NO_GPU.search(page_text(page)))


_SECRET_MISSING = re.compile(r"Colab secret (\S+) is missing or Notebook access is off|"
                             r"Secret (\S+) does not exist")


def secret_missing(page: Any) -> str | None:
    """The name of a Colab Secret the notebook could not read (names only, never values)."""
    m = _SECRET_MISSING.search(page_text(page))
    return (m.group(1) or m.group(2)) if m else None


def wait_for_runner(page: Any, log: Callable[[str], None] = print, stuck_s: int | None = None) -> list[str]:
    """Wait after Run all for the runner's first [colab-train] lines. Used on its own to resume a run already started."""
    started = time.monotonic()
    deadline = started + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if stuck_s and time.monotonic() - started > stuck_s:
            raise RunnerStuck(f"no runner output {stuck_s} s after Run all")
        browser.check_safety(page)
        if session_limit(page):
            dismissed = dismiss_session_limit(page)
            shot = browser.screenshot(page, "too-many-sessions")
            raise SessionLimit(f"too many sessions: Colab runs one session at a time on this account"
                               f"{' (Cancel clicked)' if dismissed else ''}", shot)
        if click_dialog_button(page, RUN_ANYWAY_NAMES):
            log("clicked 'Run anyway' (notebook not authored by Google)")
            continue
        if click_dialog_button(page, GRANT_ACCESS_NAMES):
            log("clicked 'Grant access' (Colab Secret for this notebook)")
            continue
        lines = output_lines(page)
        if lines:
            return lines
        if no_gpu(page):
            shot = browser.screenshot(page, "no-gpu")
            raise SafetyStop("no GPU attached to this runtime (Colab did not assign the job's GPU type). Disconnect "
                             "and delete the runtime, then try again later", shot)
        missing = secret_missing(page)
        if missing:
            shot = browser.screenshot(page, "secret-missing")
            raise SafetyStop(f"Colab Secret {missing} is missing or its Notebook access is off (add it under the key "
                             "icon, Notebook access ON, then launch again)", shot)
        page.wait_for_timeout(3000)
    raise BrowserError(f"no runner output after {START_TIMEOUT_S} s (see the screenshot)")


def disconnect_and_delete(page: Any) -> bool:
    """Connection options > Disconnect and delete runtime. Returns True when a Connect or Reconnect button is back
    (after a finished run Colab offers Reconnect instead of Connect)."""
    _open_options(page)
    page.get_by_role("menuitem", name=DISCONNECT_NAMES).first.click()
    for names in (("Yes",), ("Disconnect and delete runtime", "Disconnect and delete"), ("Delete", "OK")):
        if click_dialog_button(page, names):
            break
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if page.get_by_role("button", name=re.compile(r"^(Connect|Reconnect|Verbinden|Erneut verbinden)\b")).count():
            return True
        page.wait_for_timeout(2000)
    return False


def find_job_page(ctx: Any, saved_url: str | None, stem: str) -> Any:
    """The Colab tab of one run: the URL saved at launch, else a title containing the notebook name."""
    pages = list(ctx.pages)
    if saved_url:
        for p in pages:
            if p.url == saved_url:
                return p
    for p in pages:
        if stem and stem in (p.title() or ""):
            return p
    return None
