"""Rendered notebooks: valid nbformat, self-contained, no secret values, runnable setup cells."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from colab_train import render, spec

FAKE_SECRETS = {
    "GIT_TOKEN": "FAKE-GIT-VALUE-0001",
    "WANDB_API_KEY": "FAKE-WANDB-VALUE-0002",
    "HF_TOKEN": "FAKE-HF-VALUE-0003",
}


def validate_nbformat(nb: dict):
    try:
        import nbformat
    except ImportError:
        return  # run with: uvx --with nbformat ... to enforce this check
    nbformat.validate(nbformat.from_dict(nb))


class RenderTests(unittest.TestCase):
    def render_all(self, names):
        out = {}
        old = {k: os.environ.get(k) for k in FAKE_SECRETS}
        os.environ.update(FAKE_SECRETS)
        try:
            for n in names:
                job = spec.load(spec.PRESET_DIR / f"{n}.yaml")
                out[n] = (job, render.render_notebook(job))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return out

    def test_valid_nbformat_for_all_presets(self):
        for name, (_, nb) in self.render_all(spec.preset_names()).items():
            with self.subTest(name=name):
                validate_nbformat(nb)
                self.assertEqual(nb["nbformat"], 4)
                job_accel = spec.load_text(spec.preset_text(name))["runtime"]["accelerator"]
                self.assertEqual(nb["metadata"]["accelerator"], job_accel)

    def test_no_secret_values_anywhere(self):
        for name, (_, nb) in self.render_all(spec.preset_names()).items():
            text = json.dumps(nb)
            with self.subTest(name=name):
                for value in FAKE_SECRETS.values():
                    self.assertNotIn(value, text)
                self.assertIn("GITHUB_TOKEN" if name == "mjlab-rsl-rl" else "WANDB_API_KEY", text)

    def test_secret_names_and_runner_embedded(self):
        (job, nb), = self.render_all(["mjlab-rsl-rl"]).values()
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("WANDB_API_KEY", text)
        self.assertIn("def build_train_command", text)  # runner embedded
        self.assertIn("mjlab-colab-1", text)
        self.assertNotIn("drive.mount", text)  # wandb resume: no Drive

    def test_drive_job_mounts_drive(self):
        job = spec.load(spec.PRESET_DIR / "example-pytorch.yaml")
        job["resume"]["source"] = "drive"
        nb = render.render_notebook(job)
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("drive.mount", text)

    def test_setup_cells_run_and_write_runner_and_job(self):
        """Execute the notebook's write-files cell in a temp dir (paths rewritten), then plan with
        the written runner. Proves the notebook is self-contained."""
        (job, nb), = self.render_all(["mjlab-rsl-rl"]).values()
        with tempfile.TemporaryDirectory() as d:
            write_cell = "".join(nb["cells"][3]["source"]).replace("/content/colab-train", d)
            exec(compile(write_cell, "<cell>", "exec"), {})
            runner_path = Path(d) / "runner.py"
            job_path = Path(d) / "job.json"
            self.assertTrue(runner_path.exists() and job_path.exists())
            self.assertEqual(json.loads(job_path.read_text())["derived"]["run_id"], "mjlab-colab-1")
            buf = io.StringIO()
            import subprocess, sys
            r = subprocess.run([sys.executable, str(runner_path), "plan", "--job", str(job_path),
                                "--workdir", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Mjlab-Your-Task-Id-v0", r.stdout)

    def test_runner_source_is_stdlib_only(self):
        src = render.RUNNER_PATH.read_text()
        for mod in ("import yaml", "from colab_train", "from . "):
            self.assertNotIn(mod, src)


if __name__ == "__main__":
    unittest.main()
