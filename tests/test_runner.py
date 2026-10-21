"""Runner: command expansion, the train command of the example preset, Drive sync and restore, no token leaks."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from colab_train import runner, spec


def job_from(preset_or_dict):
    if isinstance(preset_or_dict, Path):
        return spec.load(preset_or_dict)
    job, errors = spec.validate(preset_or_dict)
    assert not errors, errors
    return job


GENERIC = {
    "name": "gen",
    "repo": {"url": "https://github.com/acme/gen.git"},
    "wandb": {"mode": "disabled"},
    "train": {"command": "python train.py --steps {iters} --run-name {run_id}{resume_args}",
              "max_iters": 10, "save_every": 2},
    "resume": {"source": "drive", "checkpoint_glob": "checkpoints/**/ckpt_*.pt",
               "checkpoint_regex": r"ckpt_(\d+)\.pt", "restore_dir": "checkpoints/{run_name}",
               "iters_mode": "total", "args": "--resume {resume_path}"},
}


class ExpandTests(unittest.TestCase):
    def test_expand_known_and_unknown(self):
        self.assertEqual(runner.expand("a {x} b", {"x": 1}), "a 1 b")
        with self.assertRaises(KeyError):
            runner.expand("{missing}", {})

    def test_shell_braces_are_not_placeholders_when_not_lowercase_names(self):
        # ${HOME} is not matched by the lowercase placeholder pattern
        self.assertEqual(runner.expand("echo ${HOME}", {}), "echo ${HOME}")


class PresetCommandTests(unittest.TestCase):
    """The mjlab-rsl-rl example must expand to the rsl_rl arguments it documents."""

    def setUp(self):
        self.job = job_from(spec.PRESET_DIR / "mjlab-rsl-rl.yaml")
        self.repo = Path("/content/colab-train/your-robot-repo")

    def test_fresh_command(self):
        cmd = runner.build_train_command(self.job, self.repo, None)
        self.assertEqual(cmd[:3], ["uv", "run", "--frozen"])
        self.assertIn("Mjlab-Your-Task-Id-v0", cmd)
        self.assertEqual(cmd[cmd.index("--agent.max-iterations") + 1], "3000")
        self.assertEqual(cmd[cmd.index("--agent.run-name") + 1], "mjlab-colab-1")
        self.assertEqual(cmd[cmd.index("--agent.wandb-project") + 1], "mjlab-rsl-rl")
        self.assertEqual(cmd[cmd.index("--agent.save-interval") + 1], "100")
        self.assertEqual(cmd[cmd.index("--video-interval") + 1], str(500 * 24))
        self.assertNotIn("--agent.resume", cmd)

    def test_resume_command_uses_remaining_iterations(self):
        res = runner.Resume(run_name="2026-01-01_00-00-00_mjlab-colab-1", checkpoint="model_199.pt",
                            iteration=199, path=Path("/x/model_199.pt"))
        cmd = runner.build_train_command(self.job, self.repo, res)
        self.assertEqual(cmd[cmd.index("--agent.max-iterations") + 1], str(3000 - 199))
        self.assertEqual(cmd[cmd.index("--agent.resume") + 1], "True")
        self.assertEqual(cmd[cmd.index("--agent.load-run") + 1], "2026-01-01_00-00-00_mjlab-colab-1")
        self.assertEqual(cmd[cmd.index("--agent.load-checkpoint") + 1], "model_199.pt")

    def test_video_off_when_gl_disabled(self):
        job = json.loads(json.dumps(self.job))
        job["video"]["every"] = 0
        cmd = runner.build_train_command(job, self.repo, None)
        self.assertNotIn("--video", cmd)

    def test_bench_template_filled(self):
        # The preset leaves bench.command empty; this is the commented example from it.
        self.job["bench"]["command"] = ("uv run --frozen eval-your-task --checkpoint {checkpoint} --task {task} "
                                        "--trials 100 --json {bench_json} --device cuda:0 "
                                        "--log-to-wandb {wandb_entity}/{wandb_project}/{run_id}-bench")
        out = runner.build_bench_command(self.job, self.repo, Path("/c/model_5.pt"), Path("/c/b.json"))
        self.assertEqual(out[out.index("--checkpoint") + 1], "/c/model_5.pt")
        self.assertIn("your-wandb-entity/mjlab-rsl-rl/mjlab-colab-1-bench", out)


class TokenHygieneTests(unittest.TestCase):
    def test_git_helper_reads_token_from_env_only(self):
        job = job_from(spec.PRESET_DIR / "mjlab-rsl-rl.yaml")
        prefix = runner.git_prefix(job)
        joined = " ".join(prefix)
        self.assertIn("$COLAB_TRAIN_GIT_TOKEN", joined)
        self.assertIn("username=x-access-token", joined)
        fake = "SECRETVALUE-123456"
        old = os.environ.get("GIT_TOKEN")
        os.environ["GIT_TOKEN"] = fake
        try:
            self.assertNotIn(fake, " ".join(prefix))
        finally:
            if old is None:
                os.environ.pop("GIT_TOKEN", None)
            else:
                os.environ["GIT_TOKEN"] = old

    def test_no_helper_for_public_repo(self):
        job = job_from(GENERIC)
        self.assertEqual(runner.git_prefix(job), ["git"])


class DriveTests(unittest.TestCase):
    def test_sync_then_restore_newest_checkpoint(self):
        job = job_from(GENERIC)
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "repo"
            drive = Path(d) / "drive" / "gen-1"
            for it in (2, 4, 6):
                p = repo / "checkpoints" / "run_a" / f"ckpt_{it}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * it)
            copied = runner.sync_up(job, repo, drive)
            self.assertEqual(len(copied), 3)
            self.assertEqual(runner.sync_up(job, repo, drive), [])
            # fresh machine: restore from Drive
            fresh = Path(d) / "fresh"
            res = runner.restore_drive(job, fresh, drive)
            self.assertEqual(res.iteration, 6)
            self.assertEqual(res.checkpoint, "ckpt_6.pt")
            self.assertEqual(res.run_name, "run_a")
            self.assertTrue(res.path.exists())
            self.assertFalse(list(fresh.rglob("*.tmp")))

    def test_restore_on_empty_drive(self):
        job = job_from(GENERIC)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(runner.restore_drive(job, Path(d) / "r", Path(d) / "empty"))

    def test_newest_local_checkpoint(self):
        job = job_from(GENERIC)
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            for it in (1, 10, 9):
                p = repo / "checkpoints" / f"ckpt_{it}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
            self.assertEqual(runner.newest_local_checkpoint(job, repo).name, "ckpt_10.pt")


class PlanCliTests(unittest.TestCase):
    def test_plan_prints_commands_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            job_path = Path(d) / "job.json"
            job_path.write_text(json.dumps(job_from(spec.PRESET_DIR / "mjlab-rsl-rl.yaml")))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                runner.main(["plan", "--job", str(job_path), "--workdir", d])
            text = buf.getvalue()
            self.assertIn("install: uv sync --frozen", text)
            self.assertIn("Mjlab-Your-Task-Id-v0", text)


if __name__ == "__main__":
    unittest.main()
