"""The W&B config change on a resumed run (rsl_rl writes train_cfg again) is allowed only through the runner's shim.
Uses the real wandb package in disabled mode, in a child process, so no network is needed."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from colab_train import runner

CHILD = """
import wandb
wandb.init(mode="disabled")
wandb.config.update({"train_cfg": {"max_iterations": 2000}})
wandb.config.update({"train_cfg": {"max_iterations": 1700, "resume": True}})
print("resumed update ok")
print("commit", wandb.config.get("colab_train_git_commit"))
"""

FRESH_CHILD = """
import wandb
wandb.init(mode="disabled")
wandb.config.update({"train_cfg": {"max_iterations": 2000}})
print("commit", wandb.config.get("colab_train_git_commit"))
try:
    wandb.config.update({"train_cfg": {"max_iterations": 1700}})
    print("change allowed")
except Exception:
    print("change refused")
"""


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_wandb_available(), "wandb is not installed")
class ResumeShimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "child.py"
        self.script.write_text(CHILD)
        self.env = {**os.environ, "WANDB_SILENT": "true"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_plain_wandb_refuses_the_resume_config_change(self):
        out = subprocess.run([sys.executable, str(self.script)], env={**self.env, "PYTHONPATH": ""},
                             capture_output=True, text=True, timeout=120)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("allow_val_change", out.stderr)

    def test_shim_lets_the_resume_config_change_through(self):
        shim = runner.write_resume_shim(Path(self.tmp.name))
        out = subprocess.run([sys.executable, str(self.script)],
                             env={**self.env, "PYTHONPATH": str(shim), "COLAB_TRAIN_ALLOW_CONFIG_CHANGE": "1",
                                  "COLAB_TRAIN_GIT_COMMIT": "6486b99abc"},
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("resumed update ok", out.stdout)
        self.assertIn("commit 6486b99abc", out.stdout)

    def test_fresh_run_records_the_commit_and_keeps_wandb_strict(self):
        shim = runner.write_resume_shim(Path(self.tmp.name))
        script = Path(self.tmp.name) / "fresh.py"
        script.write_text(FRESH_CHILD)
        out = subprocess.run([sys.executable, str(script)],
                             env={**self.env, "PYTHONPATH": str(shim), "COLAB_TRAIN_GIT_COMMIT": "6486b99abc"},
                             capture_output=True, text=True, timeout=120)
        self.assertIn("commit 6486b99abc", out.stdout, out.stderr[-400:])
        self.assertIn("change refused", out.stdout)


if __name__ == "__main__":
    unittest.main()
