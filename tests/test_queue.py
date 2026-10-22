"""The shared Colab queue: locking under concurrent writers, state transitions with fake probes, launch decisions,
notifications, cancel. No network, no browser, no real W&B: probes and launchers are passed in."""
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from colab_train import queue as q
from colab_train import worker

NOW = "2026-10-21T12:00:00+00:00"
CFG = worker.settings({})  # max_parallel 1, stall 15 min, max_resumes 2, backoff 2 h


def item(**kw):
    base = {"id": "abc12345", "job": "/x/mjlab-rsl-rl.yaml", "name": "mjlab-rsl-rl", "run_name": "colab-1",
            "priority": 0, "status": "running", "submitted": NOW, "notify_pane": "pane-7", "workdir": None,
            "resumes": 0, "run_id": "mjlab-colab-1", "iter": None, "max_iters": 3000, "sec_it": None,
            "eta_h": None, "wandb_url": None, "last_iter": None, "last_change": NOW, "started": NOW,
            "reason": "", "tab_url": None}
    base.update(kw)
    return base


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.qpath = self.dir / "queue.json"
        self.env = mock.patch.dict(os.environ, {"COLAB_TRAIN_QUEUE": str(self.qpath),
                                                "COLAB_TRAIN_EVENTS": str(self.dir / "events.log")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class LockingTest(Base):
    def test_concurrent_writers_lose_nothing(self):
        script = textwrap.dedent("""
            import sys
            sys.path.insert(0, "__ROOT__")
            from colab_train import queue as q
            for i in range(20):
                q.add("/job" + sys.argv[1] + ".yaml", "r" + str(i), name="job" + sys.argv[1])
        """).replace("__ROOT__", str(Path(__file__).resolve().parent.parent))
        procs = [subprocess.Popen([sys.executable, "-c", script, str(n)], env={**os.environ},
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for n in range(4)]
        for p in procs:
            _, err = p.communicate(timeout=120)
            self.assertEqual(p.returncode, 0, err.decode()[-300:])
        data = json.loads(self.qpath.read_text())
        self.assertEqual(len(data["items"]), 80)
        self.assertEqual(len({it["id"] for it in data["items"]}), 80)

    def test_queue_file_is_private(self):
        q.add("/a.yaml", None, name="a")
        self.assertEqual(stat.S_IMODE(os.stat(self.qpath).st_mode), 0o600)

    def test_queue_holds_no_secret_fields(self):
        q.add("/a.yaml", None, name="a", notify_pane="p")
        text = self.qpath.read_text().lower()
        for word in ("token", "password", "api_key", "secret"):
            self.assertNotIn(word, text)


class ReconcileTest(Base):
    def test_finished_at_max_iters(self):
        ch = worker.reconcile(item(), {"run_state": "running", "iteration": 3000, "max_iters": 3000}, NOW, CFG)
        self.assertEqual(ch["status"], "finished")

    def test_wandb_crashed_is_failed(self):
        ch = worker.reconcile(item(), {"run_state": "crashed", "iteration": 40}, NOW, CFG)
        self.assertEqual(ch["status"], "failed")

    def test_runner_error_line_is_failed(self):
        ch = worker.reconcile(item(), {"run_state": "running", "iteration": 5,
                                       "log_error": "training exited with 1"}, NOW, CFG)
        self.assertEqual(ch["status"], "failed")
        self.assertIn("training exited", ch["reason"])

    def test_progress_keeps_running_and_moves_last_change(self):
        later = "2026-10-21T12:05:00+00:00"
        ch = worker.reconcile(item(last_iter=10), {"run_state": "running", "iteration": 12}, later, CFG)
        self.assertEqual(ch["status"], "running")
        self.assertEqual(ch["last_change"], later)

    def test_no_new_iteration_for_15_min_is_stalled_then_recovers(self):
        late = "2026-10-21T12:16:00+00:00"
        ch = worker.reconcile(item(last_iter=12, last_change=NOW),
                              {"run_state": "running", "iteration": 12}, late, CFG)
        self.assertEqual(ch["status"], "stalled")
        self.assertIn("STALLED", ch["event"])
        back = worker.reconcile(item(status="stalled", last_iter=12, last_change=NOW),
                                {"run_state": "running", "iteration": 13}, late, CFG)
        self.assertEqual(back["status"], "running")

    def test_no_wandb_run_for_45_min_is_failed_but_not_before(self):
        self.assertEqual(worker.reconcile(item(), {"run_state": None}, "2026-10-21T12:30:00+00:00", CFG)["status"],
                         "running")
        self.assertEqual(worker.reconcile(item(), {"run_state": None}, "2026-10-21T13:00:00+00:00", CFG)["status"],
                         "failed")

    def test_disconnect_requeues_as_resume_up_to_the_limit(self):
        ch = worker.reconcile(item(resumes=0), {"run_state": "running", "iteration": 9, "runtime_gone": True},
                              NOW, CFG)
        self.assertEqual(ch["status"], "queued")
        self.assertEqual(ch["resumes"], 1)
        ch = worker.reconcile(item(resumes=2), {"run_state": "running", "iteration": 9, "runtime_gone": True},
                              NOW, CFG)
        self.assertEqual(ch["status"], "failed")
        self.assertIn("resume limit", ch["reason"])


class DecideAndClassifyTest(Base):
    def test_slot_limit_and_priority_order(self):
        q.add("/a.yaml", None, priority=0, name="a")
        q.add("/b.yaml", None, priority=5, name="b")
        data = q.snapshot()
        nxt = worker.decide_launch(data, CFG, NOW)
        self.assertEqual(nxt["name"], "b")  # higher priority first
        data["items"][0]["status"] = "running"  # a slot is taken
        self.assertIsNone(worker.decide_launch(data, CFG, NOW))
        cfg2 = worker.settings({"max_parallel": 2})
        self.assertIsNotNone(worker.decide_launch(data, cfg2, NOW))

    def test_blocked_queue_launches_nothing_until_backoff_passes(self):
        q.add("/a.yaml", None, name="a")
        data = q.snapshot()
        data["blocked_until"] = "2026-10-21T14:00:00+00:00"
        self.assertIsNone(worker.decide_launch(data, CFG, NOW))
        self.assertIsNotNone(worker.decide_launch(data, CFG, "2026-10-21T14:01:00+00:00"))

    def test_launch_outcomes(self):
        self.assertEqual(worker.classify_launch(0, "")[0], "running")
        self.assertEqual(worker.classify_launch(3, "STOP: usage limit reached. Nothing was clicked past it.")[0],
                         "queued-blocked")
        self.assertEqual(worker.classify_launch(3, "STOP: no GPU attached to this runtime (Colab did not assign")[0],
                         "queued-blocked")
        self.assertEqual(worker.classify_launch(3, "STOP: captcha. Nothing was clicked past it.")[0], "needs-human")
        self.assertEqual(worker.classify_launch(3, "STOP: google sign-in page. Nothing")[0], "needs-human")
        self.assertEqual(worker.classify_launch(1, "error: boom")[0], "failed")
        self.assertEqual(worker.classify_launch(3, "refused: 3 browser runtime(s) recorded (a, b, c)")[0],
                         "queued-session")  # a local guard count retries soon, it does not pause the queue
        multi = "job x  run y\nerror: TimeoutError: Locator.click: Timeout 30000ms exceeded.\nCall log:\n  - waiting\n (screenshot: /s.png)"
        self.assertEqual(worker.classify_launch(1, multi)[1], "error: TimeoutError: Locator.click: Timeout 30000ms exceeded.")
        self.assertEqual(worker.classify_launch(3, "STOP: too many sessions: Colab runs one session at a time on "
                                                   "this account (Cancel clicked). Nothing was clicked past it.")[0],
                         "queued-session")


class RunOnceTest(Base):
    def test_gpu_limit_lets_a_t4_job_pass_a_waiting_l4_job(self):
        cfg = worker.settings({"max_parallel": 3, "gpu_limits": {"L4": 2}})
        gpus = {"a": "L4", "b": "L4", "c": "L4", "d": "T4"}
        data = {"items": [
            {"id": "a", "status": "running", "priority": 1, "submitted": "1", "job": "a"},
            {"id": "b", "status": "running", "priority": 1, "submitted": "2", "job": "b"},
            {"id": "c", "status": "queued", "priority": 8, "submitted": "3", "job": "c"},
            {"id": "d", "status": "queued", "priority": 5, "submitted": "4", "job": "d"}]}
        pick = worker.decide_launch(data, cfg, NOW, gpu_of=lambda it: gpus[it["id"]])
        self.assertEqual(pick["id"], "d")
        data["items"][1]["status"] = "finished"
        self.assertEqual(worker.decide_launch(data, cfg, NOW, gpu_of=lambda it: gpus[it["id"]])["id"], "c")

    def test_failed_run_gets_its_runtime_stopped(self):
        cfg = worker.settings({})
        it = q.add(str(self.dir / "k1-pickup.yaml"), "colab-1", name="k1-pickup")
        worker.run_once(cfg, probe_fn=lambda _it: {}, launcher=lambda _c: (0, ""), now=NOW)
        stopped = []
        events = worker.run_once(cfg, probe_fn=lambda _it: {"run_state": None,
                                                            "log_error": "training exited with 1"},
                                 now="2026-10-21T12:05:00+00:00",
                                 stopper=lambda item: (stopped.append(item["id"]), (0, ""))[1])
        self.assertEqual(stopped, [it["id"]])
        self.assertTrue(any("runtime stop after failed done" in e for e in events))

    def test_full_pass_launches_then_finishes_and_notifies(self):
        hook = self.dir / "hook.py"
        out = self.dir / "hook.out"
        hook.write_text(f"import sys\nopen({str(out)!r}, 'a').write(sys.argv[1] + '|' + sys.argv[2] + '\\n')\n")
        cfg = worker.settings({"notify_cmd": f"{sys.executable} {hook} {{pane}}"})
        it = q.add(str(self.dir / "mjlab-rsl-rl.yaml"), "colab-1", notify_pane="pane-9", name="mjlab-rsl-rl")

        def launcher(claimed):
            return 0, ""

        events = worker.run_once(cfg, probe_fn=lambda _it: {}, launcher=launcher, now=NOW)
        self.assertTrue(any("LAUNCHED" in e for e in events))
        rec = q.find(q.snapshot(), it["id"])
        self.assertEqual(rec["status"], "running")

        events = worker.run_once(cfg, probe_fn=lambda _it: {"run_state": "finished", "iteration": 3000,
                                                       "max_iters": 3000}, launcher=launcher,
                                 now="2026-10-21T13:00:00+00:00")
        self.assertTrue(any("FINISHED" in e for e in events))
        self.assertEqual(q.find(q.snapshot(), it["id"])["status"], "finished")
        log = (self.dir / "events.log").read_text()
        self.assertIn("LAUNCHED", log)
        self.assertIn("FINISHED", log)
        hook_lines = out.read_text().splitlines()
        self.assertTrue(all(line.startswith("pane-9|") for line in hook_lines), hook_lines)

    def test_usage_limit_blocks_the_queue_and_keeps_the_job_queued(self):
        it = q.add("/a.yaml", None, name="a")
        worker.run_once(CFG, probe_fn=lambda _it: {}, launcher=lambda _c: (3, "STOP: usage limit. Nothing"), now=NOW)
        data = q.snapshot()
        self.assertEqual(q.find(data, it["id"])["status"], "queued")
        self.assertTrue(data["blocked_until"] > NOW)
        self.assertIsNone(worker.decide_launch(data, CFG, NOW))

    def test_disconnected_job_is_relaunched_once_more_in_the_queue(self):
        it = q.add("/a.yaml", "colab-1", name="a")

        def _set(data):
            rec = q.find(data, it["id"])
            rec["status"], rec["started"] = "running", NOW

        q.update(_set)
        worker.run_once(CFG, probe_fn=lambda _it: {"run_state": "running", "iteration": 4, "runtime_gone": True},
                        launcher=None, now=NOW)
        rec = q.find(q.snapshot(), it["id"])
        self.assertEqual(rec["status"], "queued")
        self.assertEqual(rec["resumes"], 1)


class ErrorRoutingTest(Base):
    def test_failures_go_to_the_error_pane_others_to_the_job_pane(self):
        hook = self.dir / "hook.py"
        out = self.dir / "hook.out"
        hook.write_text(f"import sys\nopen({str(out)!r}, 'a').write(sys.argv[1] + '|' + sys.argv[2][:40] + '\\n')\n")
        cfg = worker.settings({"notify_cmd": f"{sys.executable} {hook} {{pane}}", "error_pane": "pane-err"})
        worker.notify(item(status="failed", notify_pane="pane-job"), "X: FAILED (boom)", cfg)
        worker.notify(item(status="running", notify_pane="pane-job"), "X: running again", cfg)
        lines = out.read_text().splitlines()
        self.assertTrue(lines[0].startswith("pane-err|"), lines)
        self.assertTrue(lines[1].startswith("pane-job|"), lines)


class CancelTest(Base):
    def test_cancel_queued_marks_it_and_missing_is_reported(self):
        it = q.add("/a.yaml", None, name="a")
        prev, _ = q.cancel(it["id"])
        self.assertEqual(prev, "queued")
        self.assertEqual(q.find(q.snapshot(), it["id"])["status"], "cancelled")
        self.assertEqual(q.cancel("nope")[0], "missing")


if __name__ == "__main__":
    unittest.main()


class TabProbeTests(unittest.TestCase):
    """Jobs with W&B off are followed through the runner's lines in their Colab tab."""

    def test_progress_done_and_errors_from_tab_lines(self):
        lines = ["[colab-train] GPU: Tesla T4", "[colab-train] progress: 200/1000 (20%)",
                 "[colab-train] primary repo failed (CalledProcessError); using the fallback repo",
                 "[colab-train] W&B resume lookup failed (ValueError); starting fresh.",
                 "[colab-train] progress: 400/1000 (40%)"]
        rep = worker.parse_tab_lines(lines)
        self.assertEqual((rep["iteration"], rep["max_iters"], rep["run_state"]), (400, 1000, "running"))
        self.assertNotIn("log_error", rep)
        self.assertEqual(worker.parse_tab_lines(lines + ["[colab-train] Done. Checkpoints are kept."])["run_state"],
                         "finished")
        bad = worker.parse_tab_lines(lines + ["RuntimeError: runner exited with status 1; see above"])
        self.assertIn("runner exited with status 1", bad["log_error"])
        self.assertEqual(worker.parse_tab_lines([]), {})

    def test_run_that_continues_another_runs_iterations_is_not_finished_early(self):
        st = worker.settings({})
        item = {"job": "j", "run_name": "r", "status": "running", "started": "2026-10-22T00:00:00+00:00"}
        first = worker.reconcile(item, {"run_state": "running", "iteration": 8998, "max_iters": 2500},
                                 "2026-10-22T00:05:00+00:00", st)
        self.assertEqual((first["status"], first["first_iter"]), ("running", 8998))
        later = worker.reconcile({**item, "first_iter": 8998}, {"run_state": "running", "iteration": 11498,
                                                                "max_iters": 2500}, "2026-10-22T01:00:00+00:00", st)
        self.assertEqual(later["status"], "finished")

    def test_wandb_job_that_dies_before_its_wandb_run_fails_from_the_tab(self):
        lines = ["Traceback (most recent call last):",
                 "FileNotFoundError: /root/.cache/data.npz is missing",
                 "RuntimeError: training exited with 1; checkpoints so far are kept"]
        self.assertEqual(worker.early_exit_from_tab(lines), {"log_error": "training exited with 1"})
        self.assertEqual(worker.early_exit_from_tab(["Traceback (most recent call last):"]), {})
        verdict = worker.reconcile({"job": "j", "run_name": "r", "status": "running",
                                   "started": "2026-10-22T00:00:00+00:00"},
                                  {"run_state": None, "log_error": "training exited with 1"},
                                  "2026-10-22T00:05:00+00:00", worker.settings({}))
        self.assertEqual(verdict["status"], "failed")

    def test_saved_tab_reads_launch_state_and_unknown_tab_is_not_gone(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".colab-train").mkdir()
            (Path(d) / ".colab-train" / "state.json").write_text(json.dumps(
                {"demo": {"notebook": "/x/.colab-train/notebooks/demo-demo-1.ipynb",
                          "browser": {"url": "https://colab.research.google.com/drive/abc"}}}))
            self.assertEqual(worker.saved_tab(d, "demo"), ("https://colab.research.google.com/drive/abc", "demo-demo-1"))
            self.assertEqual(worker.saved_tab(d, "other"), (None, ""))
        # No saved URL and no notebook name: unknown, never "gone" (that used to re-queue running jobs).
        self.assertEqual(worker.browser_tab(None, ""), (None, None))
        self.assertIsNone(worker.browser_runtime_gone({"tab_url": None}))


class ReconcileIterTests(unittest.TestCase):
    def test_probe_iteration_is_shown_as_iter(self):
        it = {"job": "j", "run_name": "r", "status": "running", "started": "2026-01-01T00:00:00+00:00", "resumes": 0}
        ch = worker.reconcile(it, {"run_state": "running", "iteration": 333, "max_iters": 2000},
                              "2026-01-01T00:05:00+00:00", {"stall_min": 15, "max_resumes": 2})
        self.assertEqual((ch["iter"], ch["status"]), (333, "running"))
