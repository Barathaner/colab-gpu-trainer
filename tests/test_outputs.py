"""Result files survive a failed upload: the runner copies them into outputs/ after every command, the notebook packs and
downloads that folder when the runner exits, and the worker keeps the download before it deletes the runtime."""
import contextlib
import io
import os
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from colab_train import render, runner, spec, worker
from colab_train.spec import SpecError

BASE = """
name: out-test
repo:
  url: https://github.com/x/K1-dribble.git
{outputs}
wandb:
  mode: offline
resume:
  source: none
train:
  command: "echo train {{run_id}}"
"""


class SpecOutputsTest(unittest.TestCase):
    def test_globs_are_accepted_and_default_is_empty(self):
        self.assertEqual(spec.load_text(BASE.format(outputs=""))["outputs"], [])
        job = spec.load_text(BASE.format(outputs='outputs: ["bench/**/*.json", "logs/*.log"]'))
        self.assertEqual(job["outputs"], ["bench/**/*.json", "logs/*.log"])

    def test_paths_outside_the_repo_are_refused(self):
        for bad in ('outputs: ["/etc/passwd"]', 'outputs: ["../x/*.json"]', "outputs: bench/*.json"):
            with self.assertRaises(SpecError, msg=bad):
                spec.load_text(BASE.format(outputs=bad))


class CollectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.out = Path(self.tmp.name) / "outputs"
        (self.repo / "bench").mkdir(parents=True)
        (self.repo / "eval" / "v17").mkdir(parents=True)
        (self.repo / "bench" / "bench_model_99.json").write_text('{"ok": 1}')
        (self.repo / "eval" / "v17" / "seeda.json").write_text('{"pass": 0.8}')
        (self.repo / "eval" / "v17" / "big.bin").write_text("x")
        self.job = spec.load_text(BASE.format(outputs='outputs: ["eval/**/*.json"]'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_bench_json_and_job_globs_are_copied_once(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = runner.collect_outputs(self.job, self.repo, self.out)
            again = runner.collect_outputs(self.job, self.repo, self.out)
        self.assertEqual(sorted(first), ["bench/bench_model_99.json", "eval/v17/seeda.json"])
        self.assertEqual(again, [])
        self.assertEqual((self.out / "eval" / "v17" / "seeda.json").read_text(), '{"pass": 0.8}')

    def test_oversized_file_is_left_out(self):
        with mock.patch.object(runner, "OUTPUT_MAX_BYTES", 3), contextlib.redirect_stdout(io.StringIO()) as log:
            copied = runner.collect_outputs(self.job, self.repo, self.out)
        self.assertEqual(copied, [])
        self.assertIn("left out of the results download", log.getvalue())


class NotebookTest(unittest.TestCase):
    def test_run_cell_downloads_results_also_on_failure(self):
        job = spec.load_text(BASE.format(outputs=""))
        cell = "".join(render.render_notebook(job)["cells"][4]["source"])
        self.assertIn("finally:", cell)
        self.assertIn("files.download(arc)", cell)
        self.assertIn(f"colab-train-results-{job['derived']['run_id']}", cell)
        compile(cell, "run-cell", "exec")  # the generated cell is valid Python


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.downloads = root / "Downloads"
        self.downloads.mkdir()
        self.qdir = root / "cache"
        self.item = {"job": "jobs/x.yaml", "started": "2026-10-22T20:00:00+00:00"}

    def tearDown(self):
        self.tmp.cleanup()

    def _archive(self, name: str, mtime: float) -> Path:
        src = Path(self.tmp.name) / "src"
        src.mkdir(exist_ok=True)
        (src / "runner.log").write_text("done\n")
        path = self.downloads / name
        with tarfile.open(path, "w:gz") as t:
            t.add(src / "runner.log", arcname="runner.log")
        os.utime(path, (mtime, mtime))
        return path

    def _fetch(self, wait_s: float = 0.0) -> str:
        with mock.patch.object(worker, "item_run_id", return_value="dribble-v17-seeda"), \
                mock.patch.object(worker, "results_dir", return_value=self.qdir / "results"):
            return worker.fetch_results(self.item, {"downloads_dir": str(self.downloads)}, wait_s=wait_s, poll_s=0.01)

    def test_download_is_moved_to_results(self):
        self._archive("colab-train-results-dribble-v17-seeda (1).tar.gz", time.time())
        line = self._fetch()
        self.assertIn("results saved to", line)
        kept = list((self.qdir / "results" / "dribble-v17-seeda").glob("*.tar.gz"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(list(self.downloads.iterdir()), [])

    def test_download_from_an_earlier_launch_is_ignored(self):
        self._archive("colab-train-results-dribble-v17-seeda.tar.gz", 1_000_000_000)  # 2001: before this launch
        self.assertIn("no results download", self._fetch())

    def test_unreadable_job_does_not_wait(self):
        with mock.patch.object(worker, "item_run_id", return_value=None):
            self.assertEqual(worker.fetch_results(self.item, {}, wait_s=60), "")


if __name__ == "__main__":
    unittest.main()
