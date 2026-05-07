from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    flags = args.flags
    if flags[:1] == ["--"]:
        flags = flags[1:]
    if not flags:
        return 0

    path = args.pyproject
    text = path.read_text()
    replacement = "mojo-flags = [" + ", ".join(toml_string(flag) for flag in flags) + "]"
    updated, count = re.subn(
        r"^mojo-flags = \[.*\]$",
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        print("expected exactly one tool.mohaus.mojo-flags entry", file=sys.stderr)
        return 1

    path.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
