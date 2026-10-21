"""Browser support without a browser: state file, DevTools liveness (fake HTTP server), safety classification,
job choice and run names. No Chrome, no Playwright, no network."""
import http.server
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from colab_train import browser, pick


class FakeDevTools(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib name
        if self.path == "/json/version":
            body = json.dumps({"Browser": "Fake/1.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # keep test output quiet
        pass


class StateAndLivenessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.state = self.dir / "browser.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_state_round_trip_and_clear(self):
        self.assertIsNone(browser.read_state(self.state))
        browser.write_state({"port": 9222, "pid": 1}, self.state)
        self.assertEqual(browser.read_state(self.state)["port"], 9222)
        browser.clear_state(self.state)
        self.assertIsNone(browser.read_state(self.state))

    def test_corrupt_or_foreign_state_is_ignored(self):
        self.state.write_text("{not json")
        self.assertIsNone(browser.read_state(self.state))
        self.state.write_text(json.dumps({"pid": 5}))  # no port
        self.assertIsNone(browser.read_state(self.state))

    def test_free_port_is_a_bindable_loopback_port(self):
        port = browser.free_port()
        self.assertTrue(1024 < port < 65536)

    def test_endpoint_and_running_port_against_fake_devtools(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), FakeDevTools)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            self.assertEqual(browser.endpoint_info(port)["Browser"], "Fake/1.0")
            browser.write_state({"port": port, "pid": 1}, self.state)
            self.assertEqual(browser.running_port(self.state), port)
        finally:
            server.shutdown()
            server.server_close()
        # the server is gone: the same state file now means "not running"
        self.assertIsNone(browser.endpoint_info(port, timeout=0.5))
        self.assertIsNone(browser.running_port(self.state))

    def test_find_browser_prefers_env_then_config_then_defaults(self):
        exe = self.dir / "chrome"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        self.assertEqual(browser.find_browser({}, env={"COLAB_TRAIN_BROWSER": str(exe)}), str(exe))
        self.assertEqual(browser.find_browser({"browser": [str(self.dir / "nope"), str(exe)]}, env={}), str(exe))
        with self.assertRaises(browser.BrowserError):
            browser.find_browser({"browser": str(self.dir / "missing")}, env={})

    def test_config_and_auto_jobs_defaults(self):
        self.assertEqual(browser.load_config(self.dir / "none.json"), {})
        cfg = self.dir / "config.json"
        cfg.write_text(json.dumps({"auto_jobs": ["my-first-job"]}))
        self.assertEqual(browser.auto_jobs(browser.load_config(cfg)), ["my-first-job"])
        self.assertEqual(browser.auto_jobs({}), [])  # no built-in priority list


class ClassifyTest(unittest.TestCase):
    def test_clean_colab_page_has_no_stop(self):
        self.assertEqual(browser.classify("https://colab.research.google.com/drive/abc", []), [])

    def test_each_stop_condition(self):
        cases = {
            "captcha": (["Please verify you are human (CAPTCHA)"], []),
            "robot check": (["Are you a robot?"], []),
            "robot check (idle)": (["Are you still there?"], []),
            "usage limit": (["Cannot connect to a GPU backend. Usage limit reached."], []),
        }
        for label, (dialogs, frames) in cases.items():
            with self.subTest(label):
                reasons = browser.classify("https://colab.research.google.com/x", dialogs, frames)
                self.assertTrue(reasons, label)

    def test_failed_backend_assignment_is_a_usage_limit_stop(self):
        for text in ["Failed to assign a backend", "Couldn't assign a GPU backend", "Cannot connect to a GPU backend"]:
            with self.subTest(text=text):
                self.assertIn("usage limit", browser.classify("https://colab.research.google.com/x", [text]))

    def test_google_sign_in_url_and_captcha_iframe(self):
        self.assertIn("google sign-in page",
                      browser.classify("https://accounts.google.com/ServiceLogin?x", []))
        self.assertIn("captcha", browser.classify(
            "https://colab.research.google.com/x", [], ["https://www.google.com/recaptcha/api2/anchor"]))

    def test_ordinary_dialog_text_does_not_stop(self):
        dialogs = ["Warning: This notebook was not authored by Google", "Grant access to GIT_TOKEN"]
        self.assertEqual(browser.classify("https://colab.research.google.com/x", dialogs), [])


class NoGpuTest(unittest.TestCase):
    def test_cpu_only_runtime_is_detected_from_the_cell_error_or_runner_line(self):
        from colab_train import colab_web
        self.assertTrue(colab_web._NO_GPU.search("FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'"))
        self.assertTrue(colab_web._NO_GPU.search("RuntimeError: no GPU attached to this runtime (nvidia-smi not found)"))
        self.assertFalse(colab_web._NO_GPU.search("[colab-train] GPU: Tesla T4, 15360 MiB"))


class PickTest(unittest.TestCase):
    PRIO = ["alpha-job", "beta-job", "gamma-job"]

    def test_auto_takes_first_available_in_priority_order(self):
        job, why = pick.choose_auto(["gamma-job", "beta-job", "omega"], self.PRIO)
        self.assertEqual(job, "beta-job")
        self.assertIn("priority", why)
        job, _ = pick.choose_auto(["gamma-job", "omega"], self.PRIO)
        self.assertEqual(job, "gamma-job")

    def test_auto_falls_back_to_first_available_and_reports_it(self):
        job, why = pick.choose_auto(["example-pytorch"], self.PRIO)
        self.assertEqual(job, "example-pytorch")
        self.assertIn("none of the priority list", why)

    def test_auto_with_nothing_is_an_error(self):
        with self.assertRaises(pick.PickError):
            pick.choose_auto([], self.PRIO)

    def test_interactive_numbered_choice(self):
        shown = []
        self.assertEqual(pick.choose_interactive(["a", "b"], ask=lambda _: "2", out=shown.append), "b")
        self.assertIn("  1. a", shown)

    def test_interactive_rejects_bad_answers(self):
        for bad in ["", "0", "3", "x"]:
            with self.subTest(bad=bad), self.assertRaises(pick.PickError):
                pick.choose_interactive(["a", "b"], ask=lambda _: bad, out=lambda _: None)

    def test_fresh_run_name_skips_used_run_ids(self):
        self.assertEqual(pick.fresh_run_name("demo", "colab-1", set()), "colab-1")
        self.assertEqual(pick.fresh_run_name("demo", "colab-1", {"demo-colab-1"}), "colab-2")
        used = {"demo-colab-1", "demo-colab-2"}
        self.assertEqual(pick.fresh_run_name("demo", "colab-1", used), "colab-3")
        self.assertEqual(pick.fresh_run_name("demo", "first", {"demo-first"}), "first-2")


if __name__ == "__main__":
    unittest.main()
