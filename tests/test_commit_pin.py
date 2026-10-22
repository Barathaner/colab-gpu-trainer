"""repo.commit: the spec accepts only a commit hash, and the runner trains that commit even when the branch has moved on.
A local git repository stands in for GitHub, so no network is needed."""
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from colab_train import runner, spec
from colab_train.spec import SpecError

BASE = """
name: pin-test
repo:
  url: https://github.com/x/K1-pickup.git
  branch: main
{commit}
wandb:
  mode: offline
resume:
  source: none
train:
  command: "echo train {{run_id}}"
"""


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


class SpecCommitTest(unittest.TestCase):
    def test_hash_is_accepted(self):
        job = spec.load_text(BASE.format(commit="  commit: 6486b99"))
        self.assertEqual(job["repo"]["commit"], "6486b99")

    def test_default_is_no_pin(self):
        self.assertIsNone(spec.load_text(BASE.format(commit=""))["repo"]["commit"])

    def test_branch_name_is_refused(self):
        with self.assertRaises(SpecError):
            spec.load_text(BASE.format(commit="  commit: main"))


class CloneCommitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        src = root / "src"
        src.mkdir()
        git("init", "-q", "-b", "main", cwd=src)
        (src / "train.txt").write_text("v1\n")
        git("add", ".", cwd=src)
        git("commit", "-q", "-m", "v1", cwd=src)
        self.v1 = git("rev-parse", "HEAD", cwd=src)
        (src / "train.txt").write_text("v2\n")
        git("commit", "-q", "-am", "v2", cwd=src)  # the branch moved on after the job was written
        self.bare = root / "K1-pickup.git"
        git("clone", "-q", "--bare", str(src), str(self.bare))
        self.work = root / "work"

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, commit: str) -> dict:
        job = spec.load_text(BASE.format(commit=f"  commit: {commit}" if commit else ""))
        job["repo"]["url"] = f"file://{self.bare}"
        return job

    def test_pinned_commit_is_checked_out(self):
        with contextlib.redirect_stdout(io.StringIO()):
            repo = runner.clone_or_update(self.job(self.v1[:7]), self.work)
            again = runner.clone_or_update(self.job(self.v1[:7]), self.work)  # a restart of the same job
        self.assertEqual((again / "train.txt").read_text(), "v1\n")
        self.assertEqual(runner.head_commit(repo), self.v1)

    def test_no_pin_follows_the_branch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            repo = runner.clone_or_update(self.job(""), self.work)
        self.assertEqual((repo / "train.txt").read_text(), "v2\n")

    def test_unknown_commit_fails(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            runner.clone_or_update(self.job("deadbeef"), self.work)


if __name__ == "__main__":
    unittest.main()
