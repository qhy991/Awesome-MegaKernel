#!/usr/bin/env python3
"""Validate the data-driven Awesome MegaKernel catalog."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from catalog_lib import (
    CATALOG_PATH,
    CHANGELOG_PATH,
    MARKDOWN_PATHS,
    README_PATHS,
    REPO_ROOT,
    load_json,
    validate_catalog,
)


def main() -> int:
    try:
        data = load_json(CATALOG_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Catalog load failed: {exc}", file=sys.stderr)
        return 1

    try:
        changelog = load_json(CHANGELOG_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Changelog load failed: {exc}", file=sys.stderr)
        return 1

    readmes: dict[str, str] = {}
    markdown_docs: list[tuple[str, str, Path]] = []
    file_errors: list[str] = []
    for name, path in README_PATHS.items():
        try:
            readmes[name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            file_errors.append(f"{name}: cannot read: {exc}")

    for path in MARKDOWN_PATHS:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            file_errors.append(f"{path.relative_to(REPO_ROOT)}: cannot read: {exc}")
            continue
        markdown_docs.append(
            (str(path.relative_to(REPO_ROOT)), text, path.parent)
        )

    errors = file_errors + validate_catalog(
        data,
        changelog=changelog,
        readmes=readmes,
        markdown_docs=markdown_docs,
    )
    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    counts = {key: len(data[key]) for key in ("papers", "repos", "readings")}
    print(
        "Validated "
        f"{counts['papers']} papers/reports, "
        f"{counts['repos']} systems/repos, and "
        f"{counts['readings']} readings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
