"""Real-browser support for `colab-train browser` and `launch --browser` / `stop --browser`.

The user's installed Chrome (or Brave) runs as a normal process with a DEDICATED profile
(~/.cache/colab-train/browser-profile, never an everyday profile) and a remote-debugging port on 127.0.0.1.
Playwright attaches over CDP (`connect_over_cdp`), because Google sign-in refuses Playwright's bundled Chromium.

Nothing here reads or types passwords, 2FA codes or secret values. The state file holds only the port, the pid and
paths. Playwright is optional: everything except the CDP session works without it.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

CACHE_DIR = Path.home() / ".cache" / "colab-train"
PROFILE_DIR = CACHE_DIR / "browser-profile"
STATE_FILE = CACHE_DIR / "browser.json"
PROJECT_DIR = Path(".colab-train")
CONFIG_FILE = PROJECT_DIR / "config.json"
SCREEN_DIR = PROJECT_DIR / "screens"
COLAB_URL = "https://colab.research.google.com/?hl=en"  # hl=en: English labels, the selectors match them
# Searched in order: a bare name is looked up on PATH, an absolute path is checked as is.
DEFAULT_BINARIES = [
    "google-chrome", "google-chrome-stable", "chromium", "brave-browser",
    "/usr/bin/google-chrome", "/usr/bin/brave-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
DEFAULT_AUTO_JOBS: list[str] = []  # no built-in priority: without auto_jobs the first valid job is picked
INSTALL_HINT = "Playwright is not installed: `uv tool install 'colab-train[browser]'` or run with `uvx --with playwright`"


class BrowserError(RuntimeError):
    """Something a human or the AI can fix by following the message."""


class SafetyStop(BrowserError):
    """Colab showed a page the tool must not get past (captcha, robot check, usage limit, sign-in)."""

    def __init__(self, reason: str, screenshot: Path | None = None):
        self.reason = reason
        self.screenshot = screenshot
        super().__init__(reason)


# ---------------------------------------------------------------- config and state

def load_config(path: Path = CONFIG_FILE) -> dict:
    """Optional project config: {"browser": "/path" | ["/path", ...], "auto_jobs": ["job", ...]}."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def auto_jobs(config: dict) -> list[str]:
    jobs = config.get("auto_jobs")
    return [str(j) for j in jobs] if isinstance(jobs, list) and jobs else list(DEFAULT_AUTO_JOBS)


def find_browser(config: dict, env: dict | None = None) -> str:
    env = os.environ if env is None else env
    if env.get("COLAB_TRAIN_BROWSER"):
        cands: list[str] = [env["COLAB_TRAIN_BROWSER"]]
    elif isinstance(config.get("browser"), str):
        cands = [config["browser"]]
    elif isinstance(config.get("browser"), list):
        cands = [str(c) for c in config["browser"]]
    else:
        cands = list(DEFAULT_BINARIES)
    for c in cands:
        p = Path(c)
        if os.sep not in c and "/" not in c and not Path(c).suffix:
            found = shutil.which(c)  # a bare name such as google-chrome
            if found:
                return found
            continue
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise BrowserError(f"no browser found (tried {', '.join(cands)}). Set \"browser\" in {CONFIG_FILE} "
                       "or COLAB_TRAIN_BROWSER to a Chrome or Brave binary")


def read_state(path: Path = STATE_FILE) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "port" in data else None


def write_state(state: dict, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def clear_state(path: Path = STATE_FILE) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def endpoint_info(port: int, timeout: float = 1.0) -> dict | None:
    """The DevTools /json/version document if a browser answers on 127.0.0.1:port, else None."""
    try:
        with _NO_PROXY.open(f"http://127.0.0.1:{port}/json/version", timeout=timeout) as r:
            data = json.loads(r.read())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def running_port(state_path: Path = STATE_FILE) -> int | None:
    st = read_state(state_path)
    if st and endpoint_info(int(st["port"])):
        return int(st["port"])
    return None


# ---------------------------------------------------------------- start (foreground)

def start(config: dict, out=print, timeout: float = 30.0) -> int:
    """Start the browser in the FOREGROUND and block until it exits. No nohup, no setsid, no daemon.

    Run it in a terminal on the desktop and leave that terminal open. Closing the window (or Ctrl+C) stops it.
    """
    port = running_port()
    if port is not None:
        out(f"already running on 127.0.0.1:{port} (state {STATE_FILE}); nothing to start")
        return 0
    binary = find_browser(config)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port()
    args = [binary, f"--user-data-dir={PROFILE_DIR}", f"--remote-debugging-port={port}",
            "--remote-debugging-address=127.0.0.1", "--no-first-run", "--no-default-browser-check"]
    proc = subprocess.Popen(args)
    state = {"port": port, "pid": proc.pid, "binary": binary, "profile": str(PROFILE_DIR),
             "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_state(state)
    try:
        deadline = time.monotonic() + timeout
        while endpoint_info(port) is None:
            if proc.poll() is not None:
                raise BrowserError(f"{binary} exited with {proc.returncode} before the debugging port came up")
            if time.monotonic() > deadline:
                raise BrowserError(f"no DevTools endpoint on 127.0.0.1:{port} after {timeout:.0f} s")
            time.sleep(0.5)
        out(f"browser ready: {binary} on 127.0.0.1:{port}, profile {PROFILE_DIR}")
        out("keep this terminal open; closing the browser window stops it. Next: colab-train browser setup")
        return proc.wait()
    finally:
        current = read_state()
        if current and current.get("pid") == proc.pid:
            clear_state()


# ---------------------------------------------------------------- CDP session

def require_playwright() -> Any:
    """The sync_playwright factory, or a BrowserError that says how to install the optional extra."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise BrowserError(INSTALL_HINT) from e
    return sync_playwright


@contextlib.contextmanager
def cdp_context(port: int | None = None) -> Iterator[Any]:
    """Attach to the running browser over CDP and yield its default browser context.

    Only pages this tool opens or identifies by URL are touched. The browser itself is not closed.
    """
    if port is None:
        port = running_port()
    if port is None:
        raise BrowserError("the browser is not running. In a desktop terminal run: colab-train browser start")
    sync_playwright = require_playwright()
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}", timeout=15000)
        except Exception as e:  # Playwright raises its own error types
            raise BrowserError(f"could not attach to the browser on 127.0.0.1:{port}: {e}") from e
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        yield ctx


# ---------------------------------------------------------------- Colab page checks (pure, testable)

_STOP_RULES: list[tuple[str, re.Pattern]] = [
    ("captcha", re.compile(r"captcha|i'?m not a robot|verify (that )?you('| a)re (a )?human|unusual traffic|"
                           r"ich bin kein roboter", re.I)),
    ("robot check", re.compile(r"are you a robot|are you still there|sind sie ein roboter|bist du ein roboter|"
                               r"sind sie noch da|bist du noch da", re.I)),
    ("usage limit", re.compile(r"cannot connect to a gpu backend|failed to assign a (gpu )?backend|"
                               r"couldn'?t assign a (gpu )?backend|usage limit|gpu quota|"
                               r"kann keine verbindung zu einem gpu-backend|nutzungslimit", re.I)),
]
_SIGNIN_URL = re.compile(r"accounts\.google\.com|ServiceLogin|signin/", re.I)
_CAPTCHA_FRAME = re.compile(r"recaptcha|hcaptcha|challenges\.cloudflare", re.I)


def classify(url: str, dialog_texts: list[str], frame_srcs: list[str] | None = None) -> list[str]:
    """Reasons to stop, from the page URL, the visible dialog texts and the iframe sources. Empty list = go on."""
    reasons: list[str] = []
    if _SIGNIN_URL.search(url or ""):
        reasons.append("google sign-in page")
    if any(_CAPTCHA_FRAME.search(s or "") for s in (frame_srcs or [])):
        reasons.append("captcha")
    texts = " \n ".join(dialog_texts or [])
    for label, rx in _STOP_RULES:
        if rx.search(texts) and label not in reasons:
            reasons.append(label)
    return reasons


_SIGNED_IN_AVATAR = ('a[href*="SignOutOptions"], [aria-label^="Google Account"], '
                     '[aria-label^="Google-Konto"], [aria-label^="Google-Account"]')


def login_state(page: Any) -> str:
    """'logged_out' when a Sign in link or button is visible, 'logged_in' when the Google account avatar is (its
    SignOutOptions link works in any language), else 'unknown'. The account name in the label is never read or printed."""
    if re.search(r"accounts\.google\.com", page.url or ""):
        return "logged_out"
    if page.get_by_role("button", name=re.compile(r"^(Sign in|Anmelden)$")).count() or \
            page.get_by_role("link", name=re.compile(r"^(Sign in|Anmelden)$")).count():
        return "logged_out"
    if page.locator(_SIGNED_IN_AVATAR).count():
        return "logged_in"
    return "unknown"


_VISIBLE_DIALOGS_JS = """() => {
  const sel = '[role=dialog],[role=alertdialog],[aria-modal=true],mwc-dialog,paper-dialog,[class*=dialog],[class*=Dialog]';
  const seen = new Set(); const out = [];
  for (const el of document.querySelectorAll(sel)) {
    if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') continue;
    if (el.getBoundingClientRect().width === 0) continue;
    if (seen.has(el)) continue;
    const t = (el.innerText || '').trim();
    if (t) { out.push(t); seen.add(el); }
  }
  return out;
}"""


def dialog_texts(page: Any) -> list[str]:
    """Text of VISIBLE dialog-like panels (role dialog, aria-modal, Colab's custom dialogs). Hidden ones are skipped:
    innerText of a hidden element still returns its text, and that must not stop a run."""
    return [t for t in page.evaluate(_VISIBLE_DIALOGS_JS) if t]


def frame_srcs(page: Any) -> list[str]:
    """Sources of VISIBLE iframes. Colab embeds hidden reCAPTCHA v3 anchors on every notebook; those are not a challenge."""
    frames = page.locator("iframe")
    out = []
    for i in range(frames.count()):
        f = frames.nth(i)
        if f.is_visible():
            out.append(f.get_attribute("src") or "")
    return out


def check_safety(page: Any) -> None:
    """Raise SafetyStop (with a screenshot) if the page shows a stop condition."""
    reasons = classify(page.url, dialog_texts(page), frame_srcs(page))
    if reasons:
        shot = screenshot(page, "safety-stop")
        raise SafetyStop(", ".join(reasons), shot)


def screenshot(page: Any, step: str) -> Path | None:
    """Save .colab-train/screens/<time>-<step>.png. Never raises: a failed screenshot must not hide the real error."""
    try:
        SCREEN_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", step.lower()).strip("-") or "step"
        path = SCREEN_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.png"
        page.screenshot(path=str(path))
        return path.resolve()
    except Exception:  # noqa: BLE001 - best effort by design
        return None
