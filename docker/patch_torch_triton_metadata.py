#!/usr/bin/env python3
import re
from importlib import metadata
from pathlib import Path


def main() -> None:
    dist = metadata.distribution("torch")
    meta = Path(dist._path) / "METADATA"
    text = meta.read_text(encoding="utf-8")
    pattern = re.compile(
        r'^Requires-Dist:\s*triton==(?P<version>[^+;\s]+)(?:\+[^;\s]+)?(?P<marker>;.*)$',
        flags=re.MULTILINE,
    )
    updated, count = pattern.subn(
        r"Requires-Dist: triton>=\g<version>\g<marker>",
        text,
        count=1,
    )
    if count == 0:
        if re.search(r'^Requires-Dist:\s*triton>=', text, flags=re.MULTILINE):
            return
        raise SystemExit(f"Did not find expected Triton dependency in {meta}")
    meta.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
