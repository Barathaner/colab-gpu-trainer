"""Local secret store: permissions, imports from fake sources, and no value in any output. Fake values only."""
import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_train import cli, secrets, secrets_panel

FAKE_GIT = "fake-git-0123456789"
FAKE_WANDB = "fake-wandb-abcdefghij"
FAKE_FILE = "fake-file-token-xyz"


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = self.dir / "cfg"
        self.env = mock.patch.dict(os.environ, {"COLAB_TRAIN_CONFIG_DIR": str(self.cfg),
                                                "COLAB_TRAIN_SECRET_BACKEND": "file"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_store_dir_is_0700_and_file_is_0600(self):
        secrets.set_value("GIT_TOKEN", FAKE_GIT)
        self.assertEqual(stat.S_IMODE(os.stat(self.cfg).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(self.cfg / "secrets.json").st_mode), 0o600)

    def test_round_trip_and_names_hide_values(self):
        secrets.set_value("GIT_TOKEN", FAKE_GIT, source="import:file")
        self.assertEqual(secrets.get("GIT_TOKEN"), FAKE_GIT)
        info = secrets.names()
        self.assertEqual(set(info), {"GIT_TOKEN"})
        self.assertNotIn(FAKE_GIT, repr(info))
        self.assertNotIn("fake-", repr(info))
        self.assertEqual(info["GIT_TOKEN"]["source"], "import:file")

    def test_rm_and_missing(self):
        secrets.set_value("WANDB_API_KEY", FAKE_WANDB)
        self.assertEqual(secrets.missing(["WANDB_API_KEY", "GIT_TOKEN"]), ["GIT_TOKEN"])
        self.assertTrue(secrets.rm("WANDB_API_KEY"))
        self.assertFalse(secrets.rm("WANDB_API_KEY"))
        self.assertIsNone(secrets.get("WANDB_API_KEY"))

    def test_bad_names_and_values_are_refused(self):
        for bad in ["1BAD", "has-dash", "", "x" * 70]:
            with self.subTest(name=bad), self.assertRaises(secrets.SecretsError):
                secrets.set_value(bad, "v")
        with self.assertRaises(secrets.SecretsError) as cm:
            secrets.set_value("GIT_TOKEN", "line1\nline2")
        self.assertNotIn("line1", str(cm.exception))

    def test_import_wandb_from_temp_netrc(self):
        netrc = self.dir / ".netrc"
        netrc.write_text(f"machine api.wandb.ai\n  login user\n  password {FAKE_WANDB}\n")
        src = secrets.import_wandb(netrc_path=netrc, env={})
        self.assertIn("netrc", src)
        self.assertEqual(secrets.get("WANDB_API_KEY"), FAKE_WANDB)
        self.assertEqual(secrets.names()["WANDB_API_KEY"]["source"], "import:netrc")

    def test_import_wandb_prefers_env_and_reports_missing(self):
        secrets.import_wandb(netrc_path=self.dir / "none", env={"WANDB_API_KEY": FAKE_WANDB})
        self.assertEqual(secrets.get("WANDB_API_KEY"), FAKE_WANDB)
        with self.assertRaises(secrets.SecretsError) as cm:
            secrets.import_wandb(netrc_path=self.dir / "none", env={})
        self.assertNotIn(FAKE_WANDB, str(cm.exception))

    def test_import_file_strips_one_line_and_never_shows_it(self):
        f = self.dir / "tok"
        f.write_text(FAKE_FILE + "\n")
        secrets.import_file(f, "GIT_TOKEN")
        self.assertEqual(secrets.get("GIT_TOKEN"), FAKE_FILE)

    def test_git_credential_fill_is_parsed_without_prompting(self):
        fake = mock.Mock(returncode=0, stdout=f"protocol=https\nhost=git.example\nusername=u\npassword={FAKE_GIT}\n")
        with mock.patch.object(secrets.subprocess, "run", return_value=fake) as run:
            secrets.import_git_credential("git.example", "GIT_TOKEN")
        self.assertEqual(run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(secrets.get("GIT_TOKEN"), FAKE_GIT)

    def test_git_credential_without_password_is_an_error(self):
        fake = mock.Mock(returncode=1, stdout="", stderr="")
        with mock.patch.object(secrets.subprocess, "run", return_value=fake):
            with self.assertRaises(secrets.SecretsError):
                secrets.import_git_credential("git.example", "GIT_TOKEN")

    def test_scrub_removes_known_values(self):
        out = secrets.scrub(f"failed with {FAKE_GIT} in it", [FAKE_GIT, None])
        self.assertNotIn(FAKE_GIT, out)
        self.assertIn("***", out)


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class CliSecretsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"COLAB_TRAIN_CONFIG_DIR": str(self.dir / "cfg"),
                                                "COLAB_TRAIN_SECRET_BACKEND": "file"})
        self.env.start()
        self.cwd = os.getcwd()
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self.cwd)
        self.env.stop()
        self.tmp.cleanup()

    def run_cli(self, argv, getpass_value=None):
        out, err = io.StringIO(), io.StringIO()
        patches = []
        if getpass_value is not None:
            patches.append(mock.patch.object(cli.getpass, "getpass", return_value=getpass_value))
            patches.append(mock.patch.object(cli.sys, "stdin", FakeTTY()))
        for p in patches:
            p.start()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(argv)
        finally:
            for p in patches:
                p.stop()
        return rc, out.getvalue() + err.getvalue()

    def test_set_with_hidden_input_and_list_never_show_the_value(self):
        rc, text = self.run_cli(["secrets", "set", "GIT_TOKEN"], getpass_value=FAKE_GIT)
        self.assertEqual(rc, 0)
        self.assertNotIn(FAKE_GIT, text)
        rc, text = self.run_cli(["secrets", "list"])
        self.assertIn("GIT_TOKEN", text)
        self.assertIn("set=yes", text)
        self.assertNotIn(FAKE_GIT, text)
        self.assertNotIn(FAKE_GIT[:6], text)

    def test_piped_set_reads_stdin_and_does_not_echo(self):
        with mock.patch.object(cli.sys, "stdin", io.StringIO(FAKE_FILE + "\n")):
            rc, text = self.run_cli(["secrets", "set", "GIT_TOKEN"])
        self.assertEqual(rc, 0)
        self.assertNotIn(FAKE_FILE, text)
        self.assertEqual(secrets.get("GIT_TOKEN"), FAKE_FILE)

    def test_import_netrc_via_cli_shows_source_only(self):
        netrc = self.dir / "netrc"
        netrc.write_text(f"machine api.wandb.ai login u password {FAKE_WANDB}\n")
        with mock.patch.object(secrets.Path, "home", return_value=self.dir), \
                mock.patch.dict(os.environ, {"WANDB_API_KEY": ""}):
            (self.dir / ".netrc").write_text(netrc.read_text())
            rc, text = self.run_cli(["secrets", "import"])
        self.assertEqual(rc, 0, text)
        self.assertIn("WANDB_API_KEY: imported", text)
        self.assertNotIn(FAKE_WANDB, text)
        self.assertEqual(secrets.get("WANDB_API_KEY"), FAKE_WANDB)

    def test_rm_removes_value(self):
        secrets.set_value("GIT_TOKEN", FAKE_GIT)
        rc, text = self.run_cli(["secrets", "rm", "GIT_TOKEN"])
        self.assertEqual(rc, 0)
        self.assertNotIn(FAKE_GIT, text)
        self.assertIsNone(secrets.get("GIT_TOKEN"))

    def test_bad_name_is_an_error_without_value(self):
        rc, text = self.run_cli(["secrets", "set", "bad-name"], getpass_value=FAKE_GIT)
        self.assertEqual(rc, 1)
        self.assertNotIn(FAKE_GIT, text)

    def test_launch_refuses_a_missing_value_and_names_the_secret(self):
        # The example job is W&B online: no network here, so say there is no running run.
        with mock.patch.object(cli.browser, "require_playwright", return_value=None), \
                mock.patch("colab_train.wandb_status.find_run", return_value=None):
            rc, text = self.run_cli(["launch", "mjlab-rsl-rl", "--browser"])
        self.assertEqual(rc, 3)
        self.assertIn("GITHUB_TOKEN", text)
        self.assertIn("colab-train secrets set GITHUB_TOKEN", text)


class PanelErrorTest(unittest.TestCase):
    def test_missing_value_is_a_text_error_before_the_panel_opens(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.dict(os.environ, {"COLAB_TRAIN_CONFIG_DIR": tmp.name}):
            page = mock.Mock()
            with self.assertRaises(secrets_panel.PanelError) as cm:
                secrets_panel.ensure_secrets(page, ["GIT_TOKEN"])
            self.assertIn("GIT_TOKEN", str(cm.exception))
            page.get_by_role.assert_not_called()


if __name__ == "__main__":
    unittest.main()
