from __future__ import annotations

import unittest

from blrpng.paths import ensure_artifact_path, resolve_in_project


class PathGuardTests(unittest.TestCase):
    def test_rejects_project_escape(self):
        with self.assertRaises(ValueError):
            resolve_in_project("../outside.pt")
        with self.assertRaises(ValueError):
            ensure_artifact_path("README.md")


if __name__ == "__main__":
    unittest.main()
