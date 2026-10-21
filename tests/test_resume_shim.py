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
                             env={**self.env, "PYTHONPATH": str(shim)}, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("resumed update ok", out.stdout)


if __name__ == "__main__":
    unittest.main()
