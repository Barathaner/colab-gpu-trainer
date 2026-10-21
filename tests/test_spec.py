"""Job spec: presets validate, bad specs are rejected with clear errors, no secret values allowed."""
import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from colab_train import spec

BASE = {
    "name": "demo-job",
    "repo": {"url": "https://github.com/acme/demo.git", "token_secret": "GITHUB_TOKEN"},
    "wandb": {"entity": "acme", "project": "demo"},
    "train": {"command": "python train.py --steps {iters} --run-name {run_id}"},
    "resume": {"source": "none"},
}


class PresetTests(unittest.TestCase):
    def test_every_preset_validates(self):
        for name in spec.preset_names():
            raw = yaml.safe_load(spec.preset_text(name))
            job, errors = spec.validate(raw)
            self.assertEqual(errors, [], name)
            self.assertIn("run_id", job["derived"])

    def test_rsl_rl_resume_options_are_neutral_by_default_and_explicit_in_the_example(self):
        generic = spec.load_text(spec.preset_text("example-pytorch"))["resume"]
        self.assertEqual(generic["iters_mode"], "total")
        self.assertEqual(generic["restore_dir"], "checkpoints/{run_name}")
        self.assertIsNone(generic["experiment_config_key"])
        self.assertFalse(generic["wandb_config_shim"])
        example = spec.load(spec.PRESET_DIR / "mjlab-rsl-rl.yaml")["resume"]
        self.assertEqual(example["iters_mode"], "remaining")
        self.assertEqual(example["experiment_config_key"], "log_dir")
        self.assertTrue(example["wandb_config_shim"])

    def test_bad_rsl_rl_resume_options_are_named(self):
        text = spec.preset_text("example-pytorch")
        for bad, key in [("experiment_config_key: [x]", "experiment_config_key"),
                         ("wandb_config_shim: maybe", "wandb_config_shim")]:
            with self.subTest(key=key), self.assertRaises(spec.SpecError) as cm:
                spec.load_text(text.replace("  iters_mode: total", f"  iters_mode: total\n  {bad}"))
            self.assertTrue(any(key in e for e in cm.exception.errors), cm.exception.errors)

    def test_mjlab_preset_run_identity(self):
        job = spec.load(spec.PRESET_DIR / "mjlab-rsl-rl.yaml")
        self.assertEqual(job["derived"]["run_id"], "mjlab-colab-1")
        self.assertEqual(job["derived"]["wandb_project"], "mjlab-rsl-rl")
        self.assertEqual(job["derived"]["wandb_entity"], "your-wandb-entity")
        self.assertEqual(job["derived"]["repo_dir_name"], "your-robot-repo")
        self.assertEqual(job["resume"]["checkpoint_regex"], r"model_(\d+)\.pt")

    def test_run_name_override_changes_run_id(self):
        job = spec.load(spec.PRESET_DIR / "mjlab-rsl-rl.yaml", run_name="colab-2")
        self.assertEqual(job["derived"]["run_id"], "mjlab-colab-2")


class ValidationTests(unittest.TestCase):
    def errors(self, **changes):
        raw = copy.deepcopy(BASE)
        for path, value in changes.items():
            node = raw
            keys = path.split("__")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            if value is None:
                node.pop(keys[-1], None)
            else:
                node[keys[-1]] = value
        return spec.validate(raw)[1]

    def test_base_is_valid(self):
        self.assertEqual(self.errors(), [])

    def test_missing_train_command(self):
        errs = self.errors(train__command=None)
        self.assertTrue(any("train.command" in e for e in errs), errs)

    def test_bad_name(self):
        self.assertTrue(any("name" in e for e in self.errors(name="Bad Name!")))

    def test_url_with_credentials_rejected(self):
        errs = self.errors(repo__url="https://user:hunter2@github.com/acme/demo.git")
        self.assertTrue(any("credentials" in e for e in errs), errs)
        self.assertNotIn("hunter2", " ".join(errs))

    def test_secret_like_env_value_rejected(self):
        errs = self.errors(env={"WANDB_KEY_X": "ghp_abcdefghijklmnopqrstuvwxyz123456"})
        self.assertTrue(any("secret" in e for e in errs), errs)

    def test_env_named_like_token_rejected(self):
        errs = self.errors(env={"MY_TOKEN": "plain"})
        self.assertTrue(any("secret_env" in e for e in errs), errs)

    def test_unknown_placeholder_rejected(self):
        errs = self.errors(train__command="python train.py {nope}")
        self.assertTrue(any("nope" in e for e in errs), errs)

    def test_offline_wandb_cannot_resume_from_wandb(self):
        errs = self.errors(wandb__mode="offline", resume__source="wandb", resume__checkpoint_glob="ck/*.pt")
        self.assertTrue(any("wandb.mode online" in e for e in errs), errs)

    def test_regex_needs_group(self):
        errs = self.errors(resume__checkpoint_glob="ck/*.pt", resume__checkpoint_regex=r"model_\d+\.pt")
        self.assertTrue(any("one group" in e for e in errs), errs)

    def test_video_needs_flags(self):
        errs = self.errors(video__every=10, video__flags="")
        self.assertTrue(any("video.flags" in e for e in errs), errs)

    def test_secret_names_must_be_names(self):
        errs = self.errors(secret_env={"OK_NAME": "not a name!"})
        self.assertTrue(any("secret_env" in e for e in errs), errs)

    def test_max_hours_bounds(self):
        self.assertTrue(any("max_hours" in e for e in self.errors(limits__max_hours=30)))


class LoadTests(unittest.TestCase):
    def test_load_file_and_yaml_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "job.yaml"
            p.write_text(yaml.safe_dump(BASE))
            job = spec.load(p)
            self.assertEqual(job["derived"]["run_id"], "demo-job-1")
            self.assertEqual(job["derived"]["wandb_project"], "demo")
            p.write_text("name: [unclosed")
            with self.assertRaises(spec.SpecError):
                spec.load(p)

    def test_resolve_preset_and_unknown(self):
        self.assertEqual(spec.resolve_job_arg("mjlab-rsl-rl").name, "mjlab-rsl-rl.yaml")
        with self.assertRaises(spec.SpecError):
            spec.resolve_job_arg("no-such-job-xyz")


if __name__ == "__main__":
    unittest.main()
