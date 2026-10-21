"""End-to-end runner on this machine: a local git repo with a fake trainer, Drive as the checkpoint store.
Covers clone, train, checkpoint sync to Drive, and a second run that restores from Drive and continues."""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_train import runner, spec

FAKE_TRAINER = '''
import argparse, pathlib
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
    if step % 2 == 0:
        (out / f"ckpt_{step}.pt").write_bytes(b"w" * step)
print("trained up to", a.steps, flush=True)
'''


class E2ETests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        src = self.root / "src"
        src.mkdir()
        (src / "train.py").write_text(FAKE_TRAINER)
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
        subprocess.run([*git, "-C", str(src), "add", "."], check=True)
        subprocess.run([*git, "-C", str(src), "commit", "-q", "-m", "fake trainer"], check=True)
        self.src = src

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, steps, source="drive"):
        raw = {
            "name": "fake",
            "repo": {"url": "https://example.invalid/fake.git", "branch": "main"},
            "wandb": {"mode": "disabled"},
            "run": {"name": "1"},
            "train": {"command": f"{sys.executable} train.py --steps {{iters}} --run-name {{run_id}}"
                                 "{resume_args}",
                      "max_iters": steps, "save_every": 2},
            "resume": {"source": source, "checkpoint_glob": "checkpoints/**/ckpt_*.pt",
                       "checkpoint_regex": r"ckpt_(\d+)\.pt", "restore_dir": "checkpoints",
                       "iters_mode": "total", "args": "--resume {resume_path}"},
            "bench": {"command": None},
            "limits": {"max_hours": 1},
        }
        job, errors = spec.validate(raw)
        self.assertEqual(errors, [])
        job["repo"]["url"] = str(self.src)  # local clone for the test; the spec itself stays https
        return job

    def run_once(self, job, drive_root, workdir):
        out = io.StringIO()
        with mock.patch.object(runner, "POLL_S", 0.05), contextlib.redirect_stdout(out):
            runner.run_job(job, workdir, drive_root)
        return out.getvalue()

    def test_fresh_then_resume_from_drive(self):
        drive = self.root / "drive"
        work1 = self.root / "w1"
        log1 = self.run_once(self.job(4), drive, work1)
        self.assertIn("No checkpoint found: starting fresh", log1)
        ckpts = sorted(p.name for p in (drive / "fake-1").rglob("ckpt_*.pt"))
        self.assertEqual(ckpts, ["ckpt_2.pt", "ckpt_4.pt"])
        self.assertIn("Done.", log1)
        # second session on a fresh machine: restore from Drive and continue to 6
        work2 = self.root / "w2"
        log2 = self.run_once(self.job(6), drive, work2)
        self.assertIn("Resuming", log2)
        self.assertIn("from ckpt_4.pt (iteration 4, Drive)", log2)
        self.assertIn("--resume", log2)
        self.assertIn("train.py --steps 6", log2)
        ckpts2 = sorted(p.name for p in (drive / "fake-1").rglob("ckpt_*.pt"))
        self.assertEqual(ckpts2, ["ckpt_2.pt", "ckpt_4.pt", "ckpt_6.pt"])
        self.assertTrue(any(work2.rglob("ckpt_4.pt")))

    def test_resume_past_max_iters_trains_nothing(self):
        drive = self.root / "drive"
        self.run_once(self.job(4), drive, self.root / "w1")
        log = self.run_once(self.job(4), drive, self.root / "w2")
        self.assertIn("already reaches max_iters", log)


if __name__ == "__main__":
    unittest.main()
