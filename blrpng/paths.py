from __future__ import annotations

import hashlib
import fcntl
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def resolve_in_project(path: str | Path, *, must_exist: bool = False) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes BL-RPNG project root: {path}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def ensure_artifact_path(path: str | Path) -> Path:
    resolved = resolve_in_project(path)
    try:
        resolved.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Generated artifacts must be under {ARTIFACT_ROOT}: {path}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_status_outside_project() -> list[str]:
    """Return dirty paths outside BL-RPNG when nested in another Git repo.

    A standalone checkout is its own Git root, so there is no outer worktree
    to audit. This retains the legacy isolation check without coupling the
    project to FNOtest or to a fixed filesystem location.
    """
    root_query = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_query.returncode != 0:
        return []
    repository_root = Path(root_query.stdout.strip()).resolve()
    if repository_root == PROJECT_ROOT.resolve():
        return []
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = f"{PROJECT_ROOT.resolve().relative_to(repository_root).as_posix()}/"
    lines: list[str] = []
    for raw in result.stdout.splitlines():
        path_field = raw[3:]
        if " -> " in path_field:
            path_field = path_field.split(" -> ", 1)[1]
        if not path_field.startswith(prefix):
            lines.append(raw)
    return lines


def write_json(path: str | Path, value: Any) -> Path:
    output = ensure_artifact_path(path)
    # A killed multi-hour runner must never leave a truncated/corrupt progress
    # file.  Keep the temporary file under the guarded artifact directory and
    # atomically replace the destination only after the complete payload has
    # reached the filesystem.
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return output


@contextmanager
def exclusive_artifact_lock(name: str) -> Iterator[Path]:
    """Prevent concurrent writers from targeting the same experiment output.

    The lock file is intentionally retained as a small diagnostic record.  The
    kernel lock, rather than file existence, determines whether a run is live,
    so interrupted jobs cannot leave a stale lock that blocks later recovery.
    """
    lock_path = ensure_artifact_path(f"artifacts/experiments/locks/{name}.lock")
    stream: TextIO = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            holder = stream.read().strip() or "holder metadata unavailable"
            raise RuntimeError(
                f"Experiment output is already locked: {name}; {holder}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_unix_seconds": time.time(),
                    "name": name,
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
