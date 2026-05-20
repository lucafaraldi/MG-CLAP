"""Small helpers for provenance metadata and reproducible result summaries."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_string(argv: list[str] | None = None) -> str:
    argv = sys.argv if argv is None else argv
    return " ".join(argv)


def get_git_commit(root: os.PathLike[str] | str) -> str | None:
    root = str(root)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def file_info(path: os.PathLike[str] | str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        if p.exists()
        else None,
        "size_bytes": p.stat().st_size if p.exists() else None,
    }


def machine_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "processor": platform.processor(),
        "pid": os.getpid(),
    }


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_version": None, "cuda_available": False, "mps_available": False}
    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
