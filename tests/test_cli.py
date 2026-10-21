"""CLI verbs with W&B and gh mocked: no network, no secrets, state kept in ./.colab-train."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_train import cli, publish, wandb_status


class FakeFile:
    def __init__(self, name, size=10):
        self.name, self.size = name, size


class FakeRun:
    def __init__(self, name="2026-10-21_11-11-42_mjlab-colab-1", state="finished", it=199):
        self.name, self.id, self.state = name, "mjlab-colab-1", state
        self.url = "https://wandb.ai/your-wandb-entity/mjlab-rsl-rl/runs/mjlab-colab-1"
        self.summary = {"_step": it, "_runtime": 3600.0, "video": {"_type": "videos"}}
        self._files = [FakeFile("model_0.pt"), FakeFile("model_100.pt"), FakeFile("model_199.pt"),
                       FakeFile("media/videos/video_16.mp4"), FakeFile("output.log")]

    def files(self, per_page=50):
        return self._files

    def history(self, keys, pandas=False, samples=500):
        return [{"_step": s, "_runtime": 10.0 * s} for s in range(0, 200, 10)]


class CliTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()


class ValidateInitTests(CliTestBase):
    def test_validate_preset_prints_resolved_train_command(self):
        code, out, _ = self.run_cli("validate", "mjlab-rsl-rl")
        self.assertEqual(code, 0)
        self.assertIn("mjlab-colab-1", out)
        self.assertIn("--agent.run-name mjlab-colab-1", out)

    def test_validate_bad_file_exit_2_and_no_secret_echo(self):
        p = Path("bad.yaml")
        p.write_text("name: bad\nrepo:\n  url: https://u:hunter2@github.com/a/b.git\n"
                     "train:\n  command: python x.py\n")
        code, out, err = self.run_cli("validate", str(p))
        self.assertEqual(code, 2)
        self.assertIn("credentials", err)
        self.assertNotIn("hunter2", out + err)

    def test_init_writes_jobs_file_from_preset_with_new_name(self):
        code, out, _ = self.run_cli("init", "my-copy", "--preset", "mjlab-rsl-rl")
        self.assertEqual(code, 0)
        text = Path("jobs/my-copy.yaml").read_text()
        self.assertIn("name: my-copy", text)
        self.assertEqual(self.run_cli("init", "my-copy", "--preset", "mjlab-rsl-rl")[0], 1)

    def test_list_shows_presets(self):
        code, out, _ = self.run_cli("list")
        self.assertIn("mjlab-rsl-rl", out)
        self.assertIn("example-pytorch", out)


class RenderLaunchTests(CliTestBase):
    def test_render_writes_notebook(self):
        code, out, _ = self.run_cli("render", "example-pytorch", "--out", "nb/x.ipynb")
        self.assertEqual(code, 0)
        nb = json.loads(Path("nb/x.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)

    def test_launch_local_saves_state_and_prints_steps(self):
        with mock.patch.object(publish, "gh_ready", return_value=False):
            code, out, _ = self.run_cli("launch", "mjlab-rsl-rl", "--publish", "auto")
        self.assertEqual(code, 0)
        self.assertNotIn("one-click URL", out)
        self.assertIn("Runtime > Run all", out)
        state = json.loads(Path(".colab-train/state.json").read_text())
        self.assertEqual(state["mjlab-rsl-rl"]["run_id"], "mjlab-colab-1")
        self.assertTrue(Path(state["mjlab-rsl-rl"]["notebook"]).exists())

    def test_launch_gist_without_gh_fails(self):
        with mock.patch.object(publish, "gh_ready", return_value=False):
            code, _, err = self.run_cli("launch", "mjlab-rsl-rl", "--publish", "gist")
        self.assertEqual(code, 1)
        self.assertIn("gh", err)

    def test_launch_gist_success_builds_colab_url(self):
        def fake_run(args, **kw):
            m = mock.Mock()
            if args[:3] == ["gh", "gist", "create"]:
                m.stdout = "https://gist.github.com/someone/0123abcd\n"
            else:
                m.stdout = "someone\n"
            m.returncode = 0
            return m
        with mock.patch.object(publish, "gh_ready", return_value=True), \
                mock.patch.object(publish.subprocess, "run", side_effect=fake_run):
            code, out, _ = self.run_cli("launch", "mjlab-rsl-rl", "--publish", "gist")
        self.assertEqual(code, 0)
        self.assertIn("https://colab.research.google.com/gist/someone/0123abcd/", out)


class WandbCommandTests(CliTestBase):
    def test_status_summarizes_run(self):
        fake_api = mock.Mock()
        with mock.patch.object(wandb_status, "_api", return_value=fake_api), \
                mock.patch.object(fake_api, "runs", return_value=iter([FakeRun()])):
            code, out, _ = self.run_cli("status", "mjlab-rsl-rl")
        self.assertEqual(code, 0)
        self.assertIn("state=finished", out)
        self.assertIn("199 / 3000", out)
        self.assertIn("s/iter", out)
        self.assertIn("newest 199", out)
        self.assertIn("videos  1", out)

    def test_status_json_and_missing_run(self):
        fake_api = mock.Mock()
        with mock.patch.object(wandb_status, "_api", return_value=fake_api), \
                mock.patch.object(fake_api, "runs", return_value=iter([])):
            code, out, _ = self.run_cli("status", "mjlab-rsl-rl", "--json")
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(out)["run"])

    def test_status_ignores_bench_runs(self):
        bench = FakeRun(name="mjlab-colab-1-bench")
        fake_api = mock.Mock()
        with mock.patch.object(wandb_status, "_api", return_value=fake_api), \
                mock.patch.object(fake_api, "runs", return_value=iter([bench])):
            code, out, _ = self.run_cli("status", "mjlab-rsl-rl", "--json")
        self.assertIsNone(json.loads(out)["run"])

    def test_logs_tails_output(self):
        fake_api = mock.Mock()
        with mock.patch.object(wandb_status, "_api", return_value=fake_api), \
                mock.patch.object(fake_api, "runs", return_value=iter([FakeRun()])), \
                mock.patch.object(wandb_status, "read_log", return_value="line A\nline B") as rl:
            code, out, _ = self.run_cli("logs", "mjlab-rsl-rl", "--lines", "5")
        self.assertEqual(code, 0)
        self.assertIn("line B", out)
        rl.assert_called_once()

    def test_stop_is_honest_and_manual(self):
        code, out, _ = self.run_cli("stop", "mjlab-rsl-rl")
        self.assertEqual(code, 0)
        self.assertIn("Disconnect and delete runtime", out)

    def test_resume_refuses_when_running(self):
        fake_api = mock.Mock()
        with mock.patch.object(wandb_status, "_api", return_value=fake_api), \
                mock.patch.object(fake_api, "runs", return_value=iter([FakeRun(state="running")])):
            code, out, _ = self.run_cli("resume", "mjlab-rsl-rl")
        self.assertIn("still running", out)

    def test_resume_points_to_saved_notebook(self):
        self.run_cli("render", "mjlab-rsl-rl")  # notebook path used by state below
        Path(".colab-train").mkdir(exist_ok=True)
        Path(".colab-train/state.json").write_text(json.dumps(
            {"mjlab-rsl-rl": {"notebook": "/x/nb.ipynb", "colab_url": "https://colab.research.google.com/gist/a/b/c.ipynb"}}))
        fake_api = mock.Mock()
        with mock.patch.object(wandb_status, "_api", return_value=fake_api), \
                mock.patch.object(fake_api, "runs", return_value=iter([FakeRun(state="crashed")])):
            code, out, _ = self.run_cli("resume", "mjlab-rsl-rl")
        self.assertIn("Run all", out)
        self.assertIn("gist/a/b/c.ipynb", out)


class PublishTests(unittest.TestCase):
    def test_gist_colab_url(self):
        url = publish.gist_colab_url(Path("/tmp/x/nb.ipynb"), "https://gist.github.com/u/abc123", "u")
        self.assertEqual(url, "https://colab.research.google.com/gist/u/abc123/nb.ipynb")

    def test_local_mode_never_calls_gh(self):
        with mock.patch.object(publish.subprocess, "run", side_effect=AssertionError("no gh")):
            self.assertEqual(publish.publish(Path("x.ipynb"), "local", "d")["mode"], "local")


if __name__ == "__main__":
    unittest.main()
