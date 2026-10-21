"""The Colab UI steps against a LOCAL FAKE PAGE, over the same CDP path the real run uses.

A fake Colab (menus, dialogs, Run all output, captcha and sign-out variants) is served from 127.0.0.1. A headless
copy of the installed Chrome is started with a temp profile and a debugging port, and browser.cdp_context() attaches
to it. Skipped cleanly when Playwright or a Chrome/Chromium binary is not available. No real Google page is touched.
"""
import http.server
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from colab_train import browser, cli, colab_web, render, secrets, secrets_panel, spec

FAKE_GIT = "fake-git-0123456789"
FAKE_WANDB = "fake-wandb-abcdefghij"

FAKE = {"login": "in", "run": "normal"}  # login: in | out ; run: normal | robot

HOME = """<!doctype html><html><head><title>Colab fake home</title>
<style>[hidden]{display:none}</style></head><body>
<button aria-label="Upload notebook" id="upload-btn">Upload notebook</button>
%(account)s
<input type="file" id="picker" accept=".ipynb" hidden>
<script>
document.getElementById('upload-btn').onclick = () => document.getElementById('picker').click();
document.getElementById('picker').onchange = () => { location.href = '/notebooks/fake-nb'; };
</script></body></html>"""

ACCOUNT_IN = ('<a href="https://accounts.google.com/SignOutOptions?hl=en" aria-label="Google Account: Test User">'
              '<img alt="" width="24" height="24"></a>')
ACCOUNT_OUT = '<a href="https://accounts.google.com/ServiceLogin" role="link">Sign in</a>'

# Same labels as the live Colab UI with hl=en: toolbar "Run all", the arrow "Additional connection options"
# with a menu, the runtime-type panel (radios, not role=dialog), and dialogs that are role=dialog.
NOTEBOOK = """<!doctype html><html><head><title>fake-nb - Colaboratory</title>
<style>[hidden]{display:none} pre{white-space:pre}</style></head><body>
<button aria-label="Run all">Run all</button>
<button aria-label="Additional connection options" id="opts-btn">v</button>
<button aria-label="Connect T4" id="connect" hidden>Connect T4</button>
<button aria-label="Secrets" id="secrets-btn">key</button>
<div id="sp" hidden>
  <h2>Secrets</h2>
  <div id="rows"></div>
  <span role="button" id="add-secret" tabindex="0">add<br>Add new secret</span>
</div>
<div role="menu" id="opts-menu" hidden>
  <div role="menuitem" id="rt-type">Change runtime type</div>
  <div role="menuitem" id="rt-disc">Disconnect and delete runtime</div>
</div>
<pre class="code">  print("[colab-train] Run fake-run (this is code, not output)")</pre>
<div id="rt-panel" hidden>
  <div role="radio" aria-checked="false" tabindex="0" id="r-cpu">CPU</div>
  <div role="radio" aria-checked="false" tabindex="0" id="r-t4">T4 GPU</div>
  <div role="radio" aria-checked="false" tabindex="0" id="r-a100">A100 GPU</div>
  <button id="cancel">Cancel</button><button id="save">Save</button>
</div>
<div id="output"></div>
<div role="dialog" id="dlg-untrusted" hidden>
  <p>Warning: This notebook was not authored by Google.</p>
  <button id="cancel-u">Cancel</button><button id="anyway">Run anyway</button>
</div>
<div role="dialog" id="dlg-secret" hidden>
  <p>Notebook requests access to the secret GITHUB_TOKEN.</p>
  <button id="grant">Grant access</button>
</div>
<div role="dialog" id="dlg-robot" hidden><p>Are you a robot?</p><button>Verify</button></div>
<div role="dialog" id="dlg-disc" hidden><p>Disconnect the runtime?</p>
  <button id="disc-ok">Disconnect and delete</button></div>
<div role="dialog" id="dlg-sessions" hidden><p>Too many sessions: You have too many active sessions. Terminate an existing session to continue.</p>
  <button id="sess-cancel">Cancel</button><button id="sess-manage">Manage sessions</button></div>
<div role="dialog" id="dlg-hidden-robot" hidden><p>Are you a robot? (hidden, must be ignored)</p></div>
<iframe src="https://www.google.com/recaptcha/api2/anchor?ar=1&amp;k=fake" hidden></iframe>
<script>
const ROBOT = %(robot)s;
const SILENT = %(silent)s;
const $ = id => document.getElementById(id);
const SESSIONS = %(sessions)s;
window.__cancelled = false; window.__manage = false;
$('sess-cancel').onclick = () => { window.__cancelled = true; $('dlg-sessions').hidden = true; };
$('sess-manage').onclick = () => { window.__manage = true; };
$('opts-btn').onclick = () => { $('opts-menu').hidden = !$('opts-menu').hidden; };
$('rt-type').onclick = () => { $('opts-menu').hidden = true; $('rt-panel').hidden = false; };
['r-cpu','r-t4','r-a100'].forEach(id => $(id).onclick = () => {
  ['r-cpu','r-t4','r-a100'].forEach(o => $(o).setAttribute('aria-checked', o === id ? 'true' : 'false')); });
$('save').onclick = () => { $('rt-panel').hidden = true; };
$('cancel').onclick = () => { $('rt-panel').hidden = true; };
$('rt-disc').onclick = () => { $('opts-menu').hidden = true; $('dlg-disc').hidden = false; };
$('disc-ok').onclick = () => { $('dlg-disc').hidden = true; $('output').innerHTML = ''; $('connect').hidden = false; };
document.querySelector('button[aria-label="Run all"]').onclick = () => {
  if (SESSIONS) { $('dlg-sessions').hidden = false; return; }
  if (ROBOT) { $('dlg-robot').hidden = false; return; }
  $('dlg-untrusted').hidden = false;
};
$('anyway').onclick = () => { $('dlg-untrusted').hidden = true; $('dlg-secret').hidden = false; };
$('grant').onclick = () => {
  $('dlg-secret').hidden = true;
  if (SILENT) return;   // the runner prints nothing (a stuck run)
  $('output').innerHTML = '<pre>[colab-train] $ git clone fake\\n[colab-train] GPU: Tesla T4, 15360 MiB\\n</pre>';
};
// Secrets panel: a row per secret, labels as the live Colab UI (Delete/Show/Toggle notebook access/Value of secret)
window.__saved = {};
function mkRow(name, value, access) {
  const row = document.createElement('div');
  row.innerHTML = '<button aria-label="Show secret &quot;' + name + '&quot; value">eye</button>' +
    '<button aria-label="Delete secret &quot;' + name + '&quot;">bin</button>' +
    '<button role="switch" aria-checked="' + (access ? 'true' : 'false') + '" aria-label="Toggle notebook access for secret &quot;' + name + '&quot;">sw</button>' +
    '<input type="password" aria-label="Value of secret &quot;' + name + '&quot;" value="' + (value || '') + '">';
  const sw = row.querySelector('[role=switch]');
  sw.onclick = () => sw.setAttribute('aria-checked', sw.getAttribute('aria-checked') === 'true' ? 'false' : 'true');
  row.querySelector('[aria-label^="Delete"]').onclick = () => row.remove();
  $('rows').appendChild(row);
}
mkRow('OTHER', 'other-value', true);
$('secrets-btn').onclick = () => { $('sp').hidden = !$('sp').hidden; };
$('add-secret').onclick = () => {
  // as the live UI: a name box and a value textarea; the row is saved when the value is typed
  const row = document.createElement('div');
  row.innerHTML = '<input type="text" aria-label="Name of new secret"><textarea aria-label="Value of new secret"></textarea>';
  $('rows').appendChild(row);
  const nm = row.querySelector('input[type=text]'), vl = row.querySelector('textarea');
  vl.addEventListener('input', () => {
    if (!nm.value || window.__saved[nm.value] !== undefined) return;
    window.__saved[nm.value] = vl.value;   // the test reads this; the tool never does
    row.remove();
    mkRow(nm.value, '', false);            // saved row: Delete button, access toggle, value field
  });
};
</script></body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib name
        if self.path in ("/", ""):
            account = ACCOUNT_IN if FAKE["login"] == "in" else ACCOUNT_OUT
            body = HOME % {"account": account}
        elif self.path.startswith("/notebooks/fake-nb"):
            body = NOTEBOOK % {"robot": "true" if FAKE["run"] == "robot" else "false",
                               "silent": "true" if FAKE["run"] == "silent" else "false",
                               "sessions": "true" if FAKE["run"] == "sessions" else "false"}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def _chrome_binary():
    for cand in [os.environ.get("COLAB_TRAIN_BROWSER", ""), "/usr/bin/google-chrome", "/usr/bin/chromium",
                 "/usr/bin/chromium-browser", "/usr/bin/brave-browser"]:
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return shutil.which("chromium") or shutil.which("google-chrome")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeColabBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            browser.require_playwright()
        except browser.BrowserError as e:
            raise unittest.SkipTest(str(e))
        cls.chrome = _chrome_binary()
        if cls.chrome is None:
            raise unittest.SkipTest("no Chrome/Chromium binary found")
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.profile = tempfile.mkdtemp(prefix="ct-profile-")
        cls.port = _free_port()
        cls.proc = subprocess.Popen(
            [cls.chrome, "--headless=new", "--no-sandbox", "--no-first-run", "--disable-gpu",
             f"--user-data-dir={cls.profile}", f"--remote-debugging-port={cls.port}",
             "--remote-debugging-address=127.0.0.1", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 30
        while browser.endpoint_info(cls.port) is None:
            if cls.proc.poll() is not None or time.monotonic() > deadline:
                cls.tearDownClass()
                raise unittest.SkipTest("headless Chrome did not start with a debugging port")
            time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "proc", None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if getattr(cls, "server", None):
            cls.server.shutdown()
            cls.server.server_close()
        if getattr(cls, "profile", None):
            shutil.rmtree(cls.profile, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        FAKE.update(login="in", run="normal")
        self.url_patch = mock.patch.object(browser, "COLAB_URL", self.base + "/")
        self.url_patch.start()
        # the CLI path finds the browser through the state file; point it at the test Chrome, never the real cache
        self.port_patch = mock.patch.object(browser, "running_port", return_value=type(self).port)
        self.port_patch.start()
        # a local store with fake values, in a temp dir; the Secrets panel is filled from it
        self.secret_env = mock.patch.dict(os.environ, {"COLAB_TRAIN_CONFIG_DIR": self.tmp.name + "/cfg",
                                                       "COLAB_TRAIN_SECRET_BACKEND": "file"})
        self.secret_env.start()
        secrets.set_value("GITHUB_TOKEN", FAKE_GIT)
        secrets.set_value("WANDB_API_KEY", FAKE_WANDB)

    def tearDown(self):
        with browser.cdp_context(type(self).port) as ctx:  # close the fake tabs this test opened
            for p in list(ctx.pages):
                if p.url.startswith(self.base):
                    p.close()
        self.url_patch.stop()
        self.port_patch.stop()
        self.secret_env.stop()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_login_state_signed_in_and_out(self):
        with browser.cdp_context(self.port) as ctx:
            FAKE["login"] = "in"
            page = colab_web.open_home(ctx)
            self.assertEqual(browser.login_state(page), "logged_in")
            page.close()
            FAKE["login"] = "out"
            page = colab_web.open_home(ctx)
            self.assertEqual(browser.login_state(page), "logged_out")
            page.close()

    def test_upload_t4_run_all_and_disconnect(self):
        nb = Path(self.tmp.name) / "fake-nb.ipynb"
        nb.write_text("{}")
        with browser.cdp_context(self.port) as ctx:
            home = colab_web.open_home(ctx)
            page = colab_web.upload_notebook(home, nb)
            self.assertIn("/notebooks/fake-nb", page.url)
            self.assertEqual(colab_web.ensure_t4(page), "T4 GPU")
            logs = []
            lines = colab_web.run_all(page, log=logs.append)
            self.assertTrue(any("Run anyway" in m for m in logs))
            self.assertTrue(any("Grant access" in m for m in logs))
            self.assertTrue(any(ln.startswith("[colab-train] GPU: Tesla T4") for ln in lines))
            # the code cell mentions [colab-train] but it is not runner output
            self.assertTrue(all(ln.startswith("[colab-train]") for ln in lines))
            self.assertTrue(colab_web.disconnect_and_delete(page))
            page.close()

    def test_code_cell_marker_is_not_output(self):
        with browser.cdp_context(self.port) as ctx:
            page = self._open_notebook(ctx)
            self.assertEqual(colab_web.output_lines(page), [])
            page.close()

    def test_captcha_style_robot_check_stops_run_all(self):
        FAKE["run"] = "robot"
        with browser.cdp_context(self.port) as ctx:
            page = self._open_notebook(ctx)
            with self.assertRaises(browser.SafetyStop) as cm:
                colab_web.run_all(page, log=lambda _m: None)
            self.assertEqual(cm.exception.reason, "robot check")
            self.assertTrue(Path(cm.exception.screenshot).is_file())
            page.close()

    def test_find_job_page_by_saved_url_then_title(self):
        with browser.cdp_context(self.port) as ctx:
            page = self._open_notebook(ctx)
            found = colab_web.find_job_page(ctx, page.url, "nope")
            self.assertEqual(found.url, page.url)
            self.assertIsNotNone(colab_web.find_job_page(ctx, None, "fake-nb"))
            self.assertIsNone(colab_web.find_job_page(ctx, None, "no-such-notebook"))
            page.close()

    def _open_notebook(self, ctx):
        nb = Path(self.tmp.name) / "fake-nb.ipynb"
        nb.write_text("{}")
        home = colab_web.open_home(ctx)
        return colab_web.upload_notebook(home, nb)  # after the upload the home tab is the notebook tab

    def test_launch_and_stop_end_to_end_through_cli(self):
        """launch --browser drives the fake page, records the run, refuses a second one, and stop disconnects."""
        job = spec.load(spec.resolve_job_arg("mjlab-rsl-rl"), run_name="fake-1")
        nb = Path(".colab-train/notebooks/mjlab-rsl-rl-fake.ipynb")
        nb.parent.mkdir(parents=True)
        render.write_notebook(job, nb)
        out = cli._drive_launch(job, nb.resolve())
        self.assertEqual(out, 0)
        state = cli._state()["mjlab-rsl-rl"]
        self.assertTrue(state["browser_running"])
        self.assertEqual(state["browser"]["accelerator"], "T4 GPU")
        self.assertIn("/notebooks/fake-nb", state["browser"]["url"])
        self.assertTrue(list(Path(".colab-train/screens").glob("*.png")))
        args = type("A", (), {"job": "mjlab-rsl-rl", "run_name": None, "yes": True})()
        self.assertEqual(cli._stop_browser(args), 0)
        self.assertFalse(cli._state()["mjlab-rsl-rl"]["browser_running"])

    def _notebook_page(self, ctx):
        nb = Path(self.tmp.name) / "fake-nb.ipynb"
        nb.write_text("{}")
        return colab_web.upload_notebook(colab_web.open_home(ctx), nb)

    def test_panel_adds_missing_secrets_and_turns_access_on(self):
        logs = []
        with browser.cdp_context(self.port) as ctx:
            page = self._notebook_page(ctx)
            with mock.patch.object(browser, "screenshot") as shot:
                done = secrets_panel.ensure_secrets(page, ["GITHUB_TOKEN", "WANDB_API_KEY"], log=logs.append)
            shot.assert_not_called()  # no screenshot while the panel is open
            self.assertEqual(done, {"GITHUB_TOKEN": "added, notebook access on",
                                    "WANDB_API_KEY": "added, notebook access on"})
            # the fake records what was typed into the page: the value went from the store to the page
            self.assertEqual(page.evaluate("() => window.__saved"),
                             {"GITHUB_TOKEN": FAKE_GIT, "WANDB_API_KEY": FAKE_WANDB})
            # the other secret in the account is left alone (still on, value untouched)
            other = page.get_by_role("switch", name='Toggle notebook access for secret "OTHER"', include_hidden=True)
            self.assertEqual(other.get_attribute("aria-checked"), "true")  # panel is closed now: hidden lookup
            # nothing with a value went to the log
            text = "\n".join(logs)
            self.assertNotIn(FAKE_GIT, text)
            self.assertNotIn(FAKE_WANDB, text)
            again = secrets_panel.ensure_secrets(page, ["GITHUB_TOKEN"], log=lambda _m: None)
            self.assertEqual(again, {"GITHUB_TOKEN": "already in Colab, notebook access on"})
            page.close()

    def test_panel_update_overwrites_only_with_update_flag(self):
        with browser.cdp_context(self.port) as ctx:
            page = self._notebook_page(ctx)
            secrets_panel.ensure_secrets(page, ["GITHUB_TOKEN"], log=lambda _m: None)
            # a second value in the store; without --update the page keeps the first one
            secrets.set_value("GITHUB_TOKEN", "fake-git-second-value")
            secrets_panel.ensure_secrets(page, ["GITHUB_TOKEN"], log=lambda _m: None)
            self.assertEqual(page.evaluate("() => window.__saved.GITHUB_TOKEN"), FAKE_GIT)
            done = secrets_panel.ensure_secrets(page, ["GITHUB_TOKEN"], update=True, log=lambda _m: None)
            self.assertIn("updated", done["GITHUB_TOKEN"])
            self.assertEqual(page.get_by_label(re.compile('^Value of secret "GITHUB_TOKEN"')).first.input_value(),
                             "fake-git-second-value")
            page.close()

    def test_panel_failure_is_text_without_a_screenshot_and_without_the_value(self):
        with browser.cdp_context(self.port) as ctx:
            page = self._notebook_page(ctx)
            with mock.patch.object(browser, "screenshot") as shot, \
                    mock.patch.object(secrets_panel, "_add", side_effect=RuntimeError("boom " + FAKE_GIT)):
                with self.assertRaises(secrets_panel.PanelError) as cm:
                    secrets_panel.ensure_secrets(page, ["GITHUB_TOKEN"], log=lambda _m: None)
            shot.assert_not_called()
            self.assertNotIn(FAKE_GIT, str(cm.exception))
            self.assertIn("boom", str(cm.exception))
            page.close()

    def test_too_many_sessions_cancels_the_dialog_never_manages_and_stops(self):
        FAKE["run"] = "sessions"
        with browser.cdp_context(self.port) as ctx:
            page = self._open_notebook(ctx)
            with self.assertRaises(colab_web.SessionLimit) as cm:
                colab_web.run_all(page, log=lambda _m: None)
            self.assertIn("too many sessions", cm.exception.reason)
            self.assertIsNotNone(cm.exception.screenshot)
            self.assertTrue(page.evaluate("() => window.__cancelled"))
            self.assertFalse(page.evaluate("() => window.__manage"), "Manage sessions must never be clicked")
            page.close()

    def test_silent_runner_raises_runner_stuck(self):
        FAKE["run"] = "silent"
        with browser.cdp_context(self.port) as ctx:
            page = self._open_notebook(ctx)
            with self.assertRaises(colab_web.RunnerStuck):
                colab_web.run_all(page, log=lambda _m: None, stuck_s=3)
            page.close()

    def test_signed_out_stops_before_upload_with_screenshot(self):
        FAKE["login"] = "out"
        job = spec.load(spec.resolve_job_arg("mjlab-rsl-rl"), run_name="fake-2")
        nb = Path(".colab-train/notebooks/mjlab-rsl-rl-fake2.ipynb")
        nb.parent.mkdir(parents=True)
        render.write_notebook(job, nb)
        with self.assertRaises(browser.SafetyStop) as cm:
            cli._drive_launch(job, nb.resolve())
        self.assertIn("not signed in", cm.exception.reason)
        self.assertTrue(Path(cm.exception.screenshot).is_file())


if __name__ == "__main__":
    unittest.main()
