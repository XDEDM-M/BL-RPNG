"""Tiny stdlib test launcher for the package-free shared environment.

The prescribed environment intentionally has no third-party pytest package.
This module preserves the documented ``python -m pytest tests`` entry point
while the test suite itself uses portable ``unittest`` APIs.
"""

from __future__ import annotations

import sys
import unittest

from blrpng.paths import git_status_outside_project


def main() -> None:
    before = git_status_outside_project()
    start_dir = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), "tests")
    suite = unittest.defaultTestLoader.discover(start_dir, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = git_status_outside_project()
    if before != after:
        raise RuntimeError(
            f"Tests changed files outside BL-RPNG: before={before!r}, after={after!r}"
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
