"""repo.fallback_url: the spec accepts it only as https without credentials, and the runner clones from it only when the
primary repo fails. Local git repositories stand in for the two hosts, so no network is needed."""
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from colab_train import runner, spec
from colab_train.spec import SpecError

BASE = """
name: fb-test
repo:
  url: https://git.example.invalid/your-org/your-robot-repo.git
  branch: main
  token_secret: GIT_TOKEN
{fallback}
wandb:
  mode: offline
resume:
  source: none
train:
  command: "echo train {{run_id}}"
"""


def git(*args, cwd=None):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SpecFallbackTest(unittest.TestCase):
    def test_valid_fallback_is_accepted(self):
        job = spec.load_text(BASE.format(fallback="  fallback_url: https://github.com/your-org/your-robot-repo.git\n"
                                                  "  fallback_token_secret: GITHUB_TOKEN"))
        self.assertEqual(job["repo"]["fallback_url"], "https://github.com/your-org/your-robot-repo.git")
        self.assertEqual(job["repo"]["fallback_token_secret"], "GITHUB_TOKEN")

    def test_fallback_with_credentials_or_bad_secret_is_refused(self):
        with self.assertRaises(SpecError) as cm:
            spec.load_text(BASE.format(fallback="  fallback_url: https://user:pw@github.com/x/your-robot-repo.git"))
        self.assertNotIn("pw", str(cm.exception))
        with self.assertRaises(SpecError):
            spec.load_text(BASE.format(fallback="  fallback_url: http://github.com/x/your-robot-repo.git"))
        with self.assertRaises(SpecError):
            spec.load_text(BASE.format(fallback="  fallback_token_secret: not-a-secret-name!"))


class CredentialScopeTest(unittest.TestCase):
    def test_helper_is_offered_only_to_the_repo_host(self):
        job = spec.load_text(BASE.format(fallback="  fallback_url: https://github.com/x/your-robot-repo.git\n"
                                                  "  fallback_token_secret: GITHUB_TOKEN"))
        g = runner.git_prefix(job, "fallback", host="github.com")
        self.assertIn("credential.https://github.com.helper", g[2])
        self.assertIn("COLAB_TRAIN_GIT_TOKEN_FALLBACK", g[2])
        self.assertNotIn("COLAB_TRAIN_GIT_TOKEN\"", g[2])
        self.assertEqual(runner.git_prefix(job, "fallback", host="github.com")[0], "git")


class CloneFallbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        src = root / "src"
        src.mkdir()
        git("init", "-q", "-b", "main", cwd=src)
        (src / "train.txt").write_text("hello\n")
        git("add", ".", cwd=src)
        git("commit", "-q", "-m", "init", cwd=src)
        self.bare = root / "github-your-robot-repo.git"
        git("clone", "-q", "--bare", str(src), str(self.bare))
        self.work = root / "work"
        # the spec only takes https; the clone tests point the fallback at the local bare repo after validation
        self.job = spec.load_text(BASE.format(fallback="  fallback_url: https://github.com/x/your-robot-repo.git"))
        self.job["repo"]["fallback_url"] = f"file://{self.bare}"
        self.job["repo"]["url"] = f"file://{root}/primary-down/your-robot-repo.git"  # primary: does not exist

    def tearDown(self):
        self.tmp.cleanup()

    def test_primary_failure_uses_fallback(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repo = runner.clone_or_update(self.job, self.work)
        self.assertTrue((repo / "train.txt").is_file())
        self.assertIn("using the fallback repo", out.getvalue())

    def test_no_fallback_still_fails(self):
        self.job["repo"]["fallback_url"] = None
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            runner.clone_or_update(self.job, self.work)

    def test_existing_clone_is_switched_to_fallback_when_fetch_fails(self):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.clone_or_update(self.job, self.work)  # clone from the fallback first
        # now the primary is "up" again but the folder exists: fetch from primary fails, fallback takes over
        repo = self.work / "your-robot-repo"
        git("-C", str(repo), "remote", "set-url", "origin", f"file://{self.tmp.name}/primary-down/your-robot-repo.git")
        with contextlib.redirect_stdout(io.StringIO()):
            again = runner.clone_or_update(self.job, self.work)
        self.assertEqual(again, repo)
        self.assertTrue((repo / "train.txt").is_file())


if __name__ == "__main__":
    unittest.main()
