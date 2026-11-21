"""API keys, read from the environment or from a file outside the repository.

Both foundation models this repo compares run through hosted APIs, so two secrets are needed:
`TABPFN_TOKEN` for Prior Labs and `NIXTLA_API_KEY` for TimeGPT. Neither belongs in the repo, in a
notebook, or in a shell history. They are read from `~/.config/vn2-inventory/keys.env`, which is
outside the working tree and not tracked:

    export TABPFN_TOKEN="..."
    export NIXTLA_API_KEY="..."

`load()` never overrides a variable that is already set, so exporting a key in the shell still wins.
"""
from __future__ import annotations

import os
from pathlib import Path

CONFIG = Path.home() / ".config" / "vn2-inventory" / "keys.env"


def load(path: Path | None = None) -> None:
    """Copy `KEY=value` lines from `path` into the environment, without overwriting anything set."""
    path = CONFIG if path is None else Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def require(name: str, purpose: str) -> str:
    """Return a key, or say exactly where to put it. Called before the first API request."""
    load()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; {purpose} needs it. Put it in {CONFIG} as\n"
            f"    export {name}=\"<your key>\"\n"
            "or export it in the shell before running."
        )
    return value
