"""Repo-wide leak test: no tracked text file may contain personal or project-private words.

The words are assembled from pieces so this file does not contain them literally, and this file is skipped
by the scan. LICENSE is skipped because it carries the copyright holder's name by design.
"""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"tests/test_no_personal_data.py", "LICENSE"}

FORBIDDEN = [
    "ka" + "rl",
    "jah" + "nel",
    "ht" + "wk",
    "naos" + "erv",
    "leip" + "zig",
    "/home/" + "luna",
    "boo" + "ster",
    "core" + "weave",
    "barath" + "aner/k1-",
]


def tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    files = []
    for rel in out.decode().split("\0"):
        if not rel or rel in SKIP:
            continue
        path = ROOT / rel
        if path.is_file():
            files.append(path)
    return files


class NoPersonalDataTests(unittest.TestCase):
    def test_no_forbidden_words_in_tracked_files(self):
        try:
            files = tracked_text_files()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        hits = []
        for path in files:
            data = path.read_bytes()
            if b"\0" in data:  # binary file
                continue
            text = data.decode("utf-8", errors="ignore").lower()
            for word in FORBIDDEN:
                if word in text:
                    hits.append(f"{path.relative_to(ROOT)}: {word}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
