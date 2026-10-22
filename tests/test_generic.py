"""Generic-user coverage: example jobs for any repo, friendly validation, init, doctor, progress parsing,
and a local runner run (fake repo, fake trainer, no Colab, no network). Secret values here are fake."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_train import cli, render, runner, spec
from colab_train.spec import SpecError

FAKE_TOKEN = "FAKE-GENERIC-TOKEN-0001"
PRIVATE_WORDS = ("ht" + "wk", "ka" + "rl", "barath" + "aner", "naos" + "erv", "boo" + "ster", "k1" + "-")
GENERIC = ["example-pytorch", "cleanrl-ppo-cartpole", "sb3-ppo"]
REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(argv, cwd=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue() + err.getvalue()


class ExampleJobTests(unittest.TestCase):
    def test_generic_presets_validate_and_render(self):
        for name in GENERIC:
            with self.subTest(name=name):
                job = spec.load_text(spec.preset_text(name))
                text = json.dumps(render.render_notebook(job))
                nb = json.loads(text)
                self.assertEqual(nb["nbformat"], 4)
                self.assertEqual(len(nb["cells"]), 6)
                for word in PRIVATE_WORDS:
                    self.assertNotIn(word, text.lower(), f"private detail {word!r} leaked into {name}")

    def test_no_wandb_job_has_no_wandb_secret_or_link(self):
        job = spec.load_text(spec.preset_text("cleanrl-ppo-cartpole"))
        self.assertEqual(job["wandb"]["mode"], "disabled")
        self.assertEqual(render.secret_names(job), [])
        text = json.dumps(render.render_notebook(job))
        self.assertNotIn("wandb.ai", text)
        self.assertIn("accelerator", render.render_notebook(job)["metadata"])
        self.assertNotIn("gpuType", render.render_notebook(job)["metadata"]["colab"])
        self.assertEqual(runner.build_env(job)["WANDB_MODE"], "disabled")  # trainers that call wandb.init anyway

    def test_sb3_progress_regex_reads_logger_table(self):
        job = spec.load_text(spec.preset_text("sb3-ppo"))
        line = "| total_timesteps    | 20480    |"
        self.assertEqual(runner.parse_progress(job, line), (20480, None))
        self.assertIsNone(runner.parse_progress(job, "| time/ | fps 900 |"))

    def test_cleanrl_progress_regex_reads_global_step(self):
        job = spec.load_text(spec.preset_text("cleanrl-ppo-cartpole"))
        self.assertEqual(runner.parse_progress(job, "global_step=1024, episodic_return=22.0"), (1024, None))


class ValidationMessageTests(unittest.TestCase):
    BASE = {"name": "my-job", "repo": {"url": "https://example.invalid/me/repo.git"},
            "train": {"command": "python train.py --steps {iters}"}}

    def errors(self, **over):
        raw = json.loads(json.dumps(self.BASE))
        for k, v in over.items():
            section, key = k.split("__")
            raw.setdefault(section, {})[key] = v
        with self.assertRaises(SpecError) as cm:
            spec.load_text(json.dumps(raw))
        return cm.exception.errors

    def test_unknown_key_suggests_the_fix(self):
        errs = self.errors(train__comand="python x.py")
        self.assertTrue(any("train.comand" in e and "did you mean 'command'" in e for e in errs), errs)

    def test_missing_train_command_shows_example(self):
        raw = {"name": "my-job", "repo": {"url": "https://example.invalid/r.git"}}
        with self.assertRaises(SpecError) as cm:
            spec.load_text(json.dumps(raw))
        self.assertTrue(any(e.startswith("train.command: required") and "Example" in e for e in cm.exception.errors))

    def test_wandb_online_without_entity_names_the_fix(self):
        errs = self.errors(wandb__mode="online")
        self.assertTrue(any("wandb.entity" in e and "wandb.mode: disabled" in e for e in errs), errs)

    def test_token_value_is_refused_and_never_echoed(self):
        leaked = f"ghp_{'a' * 36}"
        with self.assertRaises(SpecError) as cm:
            spec.load_text(json.dumps({**self.BASE, "repo": {"url": "https://example.invalid/r.git",
                                                             "token_secret": leaked}}))
        text = "; ".join(cm.exception.errors)
        self.assertIn("colab-train secrets set", text)
        self.assertNotIn(leaked, text)

    def test_ssh_url_has_a_clear_message(self):
        errs = self.errors(repo__url="git@github.com:me/repo.git")
        self.assertTrue(any("repo.url" in e and "SSH" in e for e in errs), errs)

    def test_bad_progress_regex_and_gpu_type(self):
        errs = self.errors(progress__regex="step (")
        self.assertTrue(any(e.startswith("progress.regex: invalid") for e in errs), errs)
        errs = self.errors(runtime__gpu_type="V100")
        self.assertTrue(any("runtime.gpu_type" in e for e in errs), errs)

    def test_auto_defaults(self):
        job = spec.load_text(json.dumps(self.BASE))
        self.assertEqual(job["wandb"]["mode"], "disabled")
        self.assertEqual(job["resume"]["source"], "none")
        job = spec.load_text(json.dumps({**self.BASE, "wandb": {"entity": "me"},
                                         "resume": {"checkpoint_glob": "ck/*.pt"}}))
        self.assertEqual((job["wandb"]["mode"], job["resume"]["source"]), ("online", "wandb"))
        job = spec.load_text(json.dumps({**self.BASE, "resume": {"checkpoint_glob": "ck/*.pt"}}))
        self.assertEqual(job["resume"]["source"], "drive")


class ProgressAndResumeTests(unittest.TestCase):
    def test_named_and_numbered_groups(self):
        job = {"progress": {"regex": r"it (\d+) of (\d+)"}, "train": {"max_iters": 50}}
        self.assertEqual(runner.parse_progress(job, "it 7 of 50"), (7, None))
        job = {"progress": {"regex": r"step (?P<iter>\d+)/(?P<total>\d+)"}, "train": {"max_iters": 50}}
        self.assertEqual(runner.parse_progress(job, "step 12/40"), (12, 40))
        self.assertIsNone(runner.parse_progress({"progress": {"regex": None}}, "step 1"))

    def test_resume_args_and_remaining_iterations(self):
        job = spec.load_text(spec.preset_text("example-pytorch"))
        resume = runner.Resume(run_name="colab-1", checkpoint="ckpt_1500.pt", iteration=1500,
                               path=Path("/w/checkpoints/colab-1/ckpt_1500.pt"))
        ctx = runner.train_context(job, Path("/w/repo"), resume)
        self.assertEqual(ctx["iters"], job["train"]["max_iters"])  # iters_mode total
        self.assertIn("--resume /w/checkpoints/colab-1/ckpt_1500.pt", ctx["resume_args"])
        fresh = runner.train_context(job, Path("/w/repo"), None)
        self.assertEqual(fresh["resume_args"], "")

    def test_checkpoint_discovery_picks_highest_iteration(self):
        job = spec.load_text(spec.preset_text("example-pytorch"))
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for n in (500, 1500, 1000):
                d = repo / "checkpoints" / "colab-1"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"ckpt_{n}.pt").write_bytes(b"x")
            best = runner.newest_local_checkpoint(job, repo)
            self.assertEqual(best.name, "ckpt_1500.pt")


FAKE_TRAINER = '''
import argparse, pathlib, time
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, required=True)
p.add_argument("--resume", default=None)
p.add_argument("--run-name", default="x")
p.add_argument("--ckpt-dir", default="checkpoints")
a = p.parse_args()
start = 0
if a.resume:
    start = int(pathlib.Path(a.resume).stem.split("_")[1])
    print("resumed from", a.resume, flush=True)
out = pathlib.Path(a.ckpt_dir)
out.mkdir(parents=True, exist_ok=True)
for step in range(start + 1, a.steps + 1):
    print(f"step {step}/{a.steps} loss 0.1", flush=True)
    if step % 2 == 0:
        (out / f"ckpt_{step}.pt").write_bytes(b"w" * step)
print("trained up to", a.steps, flush=True)
'''


class LocalRunnerTests(unittest.TestCase):
    """runner.run_job on this machine: a local git repo as the 'remote', a fake trainer, Drive as a local dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        src = self.root / "remote" / "fake-repo"
        src.mkdir(parents=True)
        (src / "train.py").write_text(FAKE_TRAINER)
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
        subprocess.run([*git, "-C", str(src), "add", "."], check=True)
        subprocess.run([*git, "-C", str(src), "commit", "-q", "-m", "fake"], check=True)
        self.src = src

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, steps, resume_source="drive"):
        raw = {
            "name": "fake-repo",
            "repo": {"url": "https://example.invalid/me/fake-repo.git", "branch": "main"},
            "runtime": {"accelerator": "none"},
            "wandb": {"mode": "disabled"},
            "progress": {"regex": r"step (?P<iter>\d+)/(?P<total>\d+)"},
            "train": {"command": f"{sys.executable} train.py --steps {{iters}} --run-name {{run_id}}{{resume_args}}",
                      "max_iters": steps, "save_every": 2},
            "resume": {"source": resume_source, "checkpoint_glob": "checkpoints/**/ckpt_*.pt",
                       "checkpoint_regex": r"ckpt_(\d+)\.pt", "restore_dir": "checkpoints/{run_name}",
                       "iters_mode": "total", "args": "--resume {resume_path}"},
            "limits": {"max_hours": 1},
        }
        job = spec.load_text(json.dumps(raw))
        job["repo"]["url"] = f"file://{self.src}"  # the real URL is https; the clone is local
        return job

    def test_fresh_start_logs_progress_and_saves_to_drive(self):
        job = self.job(4)
        drive = self.root / "drive"
        work = self.root / "work"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runner.run_job(job, work, drive)
        text = buf.getvalue()
        self.assertIn("No checkpoint found: starting fresh.", text)
        self.assertIn("progress: 1/4 (25%)", text)
        self.assertIn("progress: 4/4 (100%)", text)
        self.assertIn("no W&B", text)
        run_dir = drive / job["derived"]["run_id"]
        self.assertTrue(any(run_dir.rglob("ckpt_4.pt")), list(run_dir.rglob("*")))

    def test_resume_from_drive_passes_resume_args(self):
        job = self.job(6)
        drive = self.root / "drive"
        run_dir = drive / job["derived"]["run_id"] / "checkpoints" / "fake-repo-1"
        run_dir.mkdir(parents=True)
        (run_dir / "ckpt_4.pt").write_bytes(b"w" * 4)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runner.run_job(job, self.root / "work2", drive)
        text = buf.getvalue()
        self.assertIn("Resuming", text)
        self.assertIn("resumed from", text)  # the fake trainer got --resume
        self.assertIn("progress: 6/6", text)

    def test_missing_gpu_refusal_names_the_gpu_type(self):
        job = self.job(2)
        job["runtime"] = {"accelerator": "GPU", "gpu_type": "L4"}
        with mock.patch.object(runner.subprocess, "run", side_effect=OSError("no nvidia-smi")):
            with self.assertRaises(RuntimeError) as cm:
                runner.run_job(job, self.root / "w3", None)
        self.assertIn("assigns a L4", str(cm.exception))
        self.assertIn("runtime.accelerator: none", str(cm.exception))


class InitAndDoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"COLAB_TRAIN_CONFIG_DIR": str(Path(self.tmp.name) / "cfg")})
        self.env.start()
        self.no_net = mock.patch.object(cli, "_remote_default_branch", return_value=None)  # init asks no network
        self.no_net.start()

    def tearDown(self):
        self.no_net.stop()
        self.env.stop()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_init_flags_write_a_valid_job(self):
        rc, text = run_cli(["init", "--yes", "--repo", "https://github.com/me/my-repo.git",
                            "--train", "python train.py --steps {iters}", "--install", "pip install -q -r requirements.txt",
                            "--progress-regex", r"step (\d+)/\d+", "--wandb", "no", "--gpu", "none"])
        self.assertEqual(rc, 0, text)
        job = spec.load(Path("jobs/my-repo.yaml"))
        self.assertEqual(job["repo"]["url"], "https://github.com/me/my-repo.git")
        self.assertEqual(job["runtime"]["accelerator"], "none")
        self.assertEqual(job["wandb"]["mode"], "disabled")
        self.assertIn("colab-train doctor", text)

    def test_init_refuses_without_train_command(self):
        rc, text = run_cli(["init", "--yes", "--repo", "https://github.com/me/my-repo.git"])
        self.assertNotEqual(rc, 0)
        self.assertIn("train.command", text)
        self.assertFalse(Path("jobs/my-repo.yaml").exists())

    def test_doctor_reports_job_errors_and_exit_code(self):
        Path("jobs").mkdir()
        Path("jobs/bad.yaml").write_text("name: bad-job\nrepo:\n  url: https://example.invalid/r.git\n"
                                         "train:\n  comand: python x.py\n")
        rc, text = run_cli(["doctor", "--no-network", "jobs/bad.yaml"])
        self.assertEqual(rc, 1)
        self.assertIn("train.comand: unknown key", text)

    def test_doctor_names_missing_secrets_without_values(self):
        from colab_train import secrets
        secrets.set_value("GITHUB_TOKEN", FAKE_TOKEN)
        Path("jobs").mkdir()
        Path("jobs/priv.yaml").write_text("name: priv-job\nrepo:\n  url: https://example.invalid/r.git\n"
                                          "  token_secret: GITHUB_TOKEN\ntrain:\n  command: python x.py\n"
                                          "wandb:\n  mode: disabled\n")
        rc, text = run_cli(["doctor", "--no-network", "jobs/priv.yaml"])
        self.assertEqual(rc, 0, text)
        self.assertIn("secret GITHUB_TOKEN set locally", text)
        self.assertNotIn(FAKE_TOKEN, text)

    def test_missing_wandb_gives_one_line_with_install_command(self):
        from colab_train import wandb_status
        Path("jobs").mkdir()
        Path("jobs/wb.yaml").write_text("name: wb-job\nrepo:\n  url: https://example.invalid/r.git\n"
                                        "train:\n  command: python x.py\nwandb:\n  mode: online\n  entity: me\n")
        with mock.patch.dict(sys.modules, {"wandb": None}):
            for verb in ("status", "logs"):
                with self.subTest(verb=verb):
                    rc, text = run_cli([verb, "jobs/wb.yaml"])
                    self.assertEqual(rc, 2, text)
                    self.assertNotIn("Traceback", text)
                    self.assertEqual(len(text.strip().splitlines()), 1, text)
                    self.assertIn(wandb_status.INSTALL_HINT, text)
            rc, text = run_cli(["doctor", "--no-network", "jobs/wb.yaml"])
        self.assertEqual(rc, 1, text)
        self.assertIn("[FAIL] wandb not installed", text)
        self.assertIn("colab-train[wandb,browser]", text)

    def test_repo_reachability_uses_head_only_and_hides_token(self):
        src = Path(self.tmp.name) / "remote"
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
        (src / "a.txt").write_text("a")
        subprocess.run([*git, "-C", str(src), "add", "."], check=True)
        subprocess.run([*git, "-C", str(src), "commit", "-q", "-m", "a"], check=True)
        job = spec.load_text(json.dumps({"name": "reach-job", "repo": {"url": "https://example.invalid/r.git"},
                                         "wandb": {"mode": "disabled"}, "train": {"command": "python x.py"}}))
        job["repo"]["url"] = f"file://{src}"
        ok, detail = cli._repo_reachable(job, None)
        self.assertTrue(ok, detail)
        self.assertNotIn(FAKE_TOKEN, detail)
        job["repo"]["url"] = f"file://{src}-missing"
        ok, detail = cli._repo_reachable(job, FAKE_TOKEN)
        self.assertFalse(ok)
        self.assertNotIn(FAKE_TOKEN, detail)
        job["repo"]["url"] = f"file://{src}"
        job["repo"]["branch"] = "master"
        ok, detail = cli._repo_reachable(job, None)
        self.assertFalse(ok)
        self.assertIn("branch 'master' not found", detail)
        self.assertIn("default branch is 'main'", detail)
        self.assertEqual(cli._default_branch(cli._ls_remote(job, None, "HEAD")[1]), "main")

    def test_status_without_wandb_reads_the_colab_tab(self):
        from colab_train import worker
        Path("jobs").mkdir()
        Path("jobs/off.yaml").write_text("name: off-job\nrepo:\n  url: https://example.invalid/r.git\n"
                                         "train:\n  command: python x.py\n  max_iters: 1000\nwandb:\n  mode: disabled\n")
        with mock.patch.object(worker, "browser_tab", return_value=(None, None)):
            rc, text = run_cli(["status", "jobs/off.yaml"])
        self.assertEqual(rc, 0, text)
        self.assertNotIn("Traceback", text)
        self.assertIn("W&B is off for this job", text)
        tab = ["[colab-train] progress: 300/1000 (30%)"]
        with mock.patch.object(worker, "browser_tab", return_value=(False, tab)):
            rc, text = run_cli(["status", "jobs/off.yaml"])
            self.assertIn("state=running", text)
            self.assertIn("300 / 1000", text)
            rc, text = run_cli(["logs", "jobs/off.yaml", "--lines", "5"])
            self.assertIn("progress: 300/1000", text)
        rc, text = run_cli(["launch", "jobs/off.yaml"])
        self.assertEqual(rc, 0, text)
        self.assertIn("no W&B", text)
        self.assertIn("Secrets: none needed", text)

    def test_init_uses_the_remote_default_branch(self):
        with mock.patch.object(cli, "_remote_default_branch", return_value="master"):
            rc, text = run_cli(["init", "--yes", "--repo", "https://github.com/me/old-repo.git",
                                "--train", "python train.py", "--wandb", "no"])
        self.assertEqual(rc, 0, text)
        self.assertEqual(spec.load(Path("jobs/old-repo.yaml"))["repo"]["branch"], "master")


if __name__ == "__main__":
    unittest.main()
