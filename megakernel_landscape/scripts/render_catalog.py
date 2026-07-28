#!/usr/bin/env python3
"""Render catalog-backed sections into both root README files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from urllib.parse import urlparse

from catalog_lib import (
    CATALOG_PATH,
    CHANGELOG_PATH,
    README_PATHS,
    load_json,
    marker_layout_errors,
    validate_catalog,
)


README_TARGETS = {
    "en": README_PATHS["README.md"],
    "zh": README_PATHS["README.zh-CN.md"],
}

SCOPE_LABELS = {
    "en": {"core": "Core", "foundation": "Foundation", "adjacent": "Adjacent"},
    "zh": {"core": "核心", "foundation": "基础", "adjacent": "相邻"},
}


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def escape_markdown_label(value: Any) -> str:
    return (
        str(value)
        .replace("\n", " ")
        .strip()
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def escape_mermaid(value: Any) -> str:
    return str(value).replace("\n", " ").replace(":", " —").strip()


def link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def github_slug(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def project_cell(entry: dict[str, Any], lang: str) -> str:
    name = escape_markdown_label(entry["name"])
    cell = link(f"**{name}**", entry["url"])
    extras: list[str] = []
    extra_labels = {
        "docs": "Docs" if lang == "en" else "文档",
        "project": "Project" if lang == "en" else "主页",
    }
    for field, label in extra_labels.items():
        extra_url = entry.get(field)
        if extra_url and extra_url != entry["url"]:
            extras.append(link(label, extra_url))
    if extras:
        cell += "<br>" + " · ".join(extras)
    slug = github_slug(entry["url"])
    if slug:
        badge = (
            f"![GitHub stars](https://img.shields.io/github/stars/{slug}"
            "?style=flat-square&label=%E2%98%85)"
        )
        cell += f"<br>{badge}"
    return cell


def categories(data: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return data["categories"][kind]


def sorted_entries(
    data: dict[str, Any], kind: str, category_id: str
) -> list[dict[str, Any]]:
    entries = [entry for entry in data[kind] if entry["category"] == category_id]
    dated = sorted(
        (entry for entry in entries if entry.get("date")),
        key=lambda item: item["name"].casefold(),
    )
    dated.sort(key=lambda item: item["date"], reverse=True)
    undated = sorted(
        (entry for entry in entries if not entry.get("date")),
        key=lambda item: item["name"].casefold(),
    )
    return dated + undated


def scope_and_summary(entry: dict[str, Any], lang: str) -> str:
    scope = SCOPE_LABELS[lang][entry["scope"]]
    summary = escape_cell(entry[f"summary_{lang}"])
    topics = entry.get("topics", [])
    escaped_topics = [escape_cell(tag).replace("`", "\\`") for tag in topics]
    topic_text = " ".join(f"`{tag}`" for tag in escaped_topics)
    suffix = f"<br><sub>{topic_text}</sub>" if topic_text else ""
    return f"**{scope}** — {summary}{suffix}"


def render_papers(data: dict[str, Any], lang: str) -> str:
    lines: list[str] = []
    headers = {
        "en": "| Work | Venue | First public | Code | Scope & contribution |",
        "zh": "| 工作 | 会议/来源 | 首次公开 | 代码 | 范围与贡献 |",
    }
    separator = "| :--- | :---: | :---: | :---: | :--- |"
    code_label = "Code" if lang == "en" else "代码"
    for category in categories(data, "papers"):
        lines.extend(
            [
                f"### {escape_markdown_label(category[f'title_{lang}'])}",
                "",
                escape_cell(category[f"note_{lang}"]),
                "",
                headers[lang],
                separator,
            ]
        )
        for entry in sorted_entries(data, "papers", category["id"]):
            title = link(
                f"**{escape_markdown_label(entry['name'])}**", entry["url"]
            )
            code = link(code_label, entry["code"]) if entry.get("code") else "—"
            lines.append(
                "| "
                + " | ".join(
                    [
                        title,
                        escape_cell(entry["venue"]),
                        entry["date"],
                        code,
                        scope_and_summary(entry, lang),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_repos(data: dict[str, Any], lang: str) -> str:
    lines: list[str] = []
    headers = {
        "en": "| Project | Target / role | Hardware & artifact status | Why it matters |",
        "zh": "| 项目 | 目标 / 角色 | 硬件与制品状态 | 价值 |",
    }
    separator = "| :--- | :--- | :--- | :--- |"
    for category in categories(data, "repos"):
        lines.extend(
            [
                f"### {escape_markdown_label(category[f'title_{lang}'])}",
                "",
                escape_cell(category[f"note_{lang}"]),
                "",
                headers[lang],
                separator,
            ]
        )
        for entry in sorted_entries(data, "repos", category["id"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        project_cell(entry, lang),
                        escape_cell(entry[f"target_{lang}"]),
                        escape_cell(entry[f"status_{lang}"]),
                        scope_and_summary(entry, lang),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_readings(data: dict[str, Any], lang: str) -> str:
    lines: list[str] = []
    kind_punctuation = "." if lang == "en" else "。"
    for category in categories(data, "readings"):
        lines.extend(
            [
                f"### {escape_markdown_label(category[f'title_{lang}'])}",
                "",
                escape_cell(category[f"note_{lang}"]),
                "",
            ]
        )
        for entry in sorted_entries(data, "readings", category["id"]):
            kind = escape_cell(entry[f"kind_{lang}"])
            name = escape_markdown_label(entry["name"])
            summary = escape_cell(entry[f"summary_{lang}"])
            lines.append(
                f"- [{name}]({entry['url']}) — "
                f"**{kind}{kind_punctuation}** {summary}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_timeline(data: dict[str, Any], lang: str) -> str:
    title = (
        "MegaKernel field timeline (curated)"
        if lang == "en"
        else "MegaKernel 领域时间线（精选）"
    )
    selected: list[dict[str, Any]] = []
    for kind in ("papers", "repos"):
        selected.extend(
            entry
            for entry in data[kind]
            if entry.get("timeline_label_en") and entry.get("timeline_label_zh")
        )
    selected.sort(key=lambda item: (item["date"], item[f"timeline_label_{lang}"]))

    grouped: dict[str, list[str]] = {}
    for entry in selected:
        grouped.setdefault(entry["date"], []).append(
            escape_mermaid(entry[f"timeline_label_{lang}"])
        )

    lines = ["```mermaid", "timeline", f"    title {title}"]
    for entry_date, labels in grouped.items():
        first, *rest = labels
        lines.append(f"    {entry_date} : {first}")
        lines.extend(f"            : {label}" for label in rest)
    lines.append("```")
    return "\n".join(lines)


def replace_block(text: str, marker: str, content: str) -> str:
    start = f"<!-- CATALOG:{marker}:START -->"
    end = f"<!-- CATALOG:{marker}:END -->"
    start_index = text.index(start)
    end_index = text.index(end, start_index + len(start))
    return (
        text[: start_index + len(start)]
        + "\n"
        + content
        + "\n"
        + text[end_index:]
    )


def replace_counts(text: str, data: dict[str, Any]) -> str:
    replacements = {
        "Papers": len(data["papers"]),
        "Systems": len(data["repos"]),
        "Readings": len(data["readings"]),
    }
    for label, count in replacements.items():
        pattern = re.compile(rf"(badge/{label}-)\d+(-[^)]+)")
        text, matches = pattern.subn(rf"\g<1>{count}\g<2>", text)
        if matches != 1:
            raise ValueError(f"expected one {label} count badge, found {matches}")
    return text


def replace_updated(text: str, data: dict[str, Any], lang: str) -> str:
    updated = data["updated"]
    badge_date = updated.replace("-", "--")
    badge_pattern = re.compile(
        r"(badge/Last%20Updated-)\d{4}--\d{2}--\d{2}(-blue\.svg)"
    )
    text, badge_matches = badge_pattern.subn(
        rf"\g<1>{badge_date}\g<2>", text
    )
    if badge_matches != 1:
        raise ValueError(
            f"expected one Last Updated badge, found {badge_matches}"
        )

    labels = {
        "en": r"(\*\*Last curated:\*\* )\d{4}-\d{2}-\d{2}(\.)",
        "zh": r"(\*\*最近整理：\*\*)\d{4}-\d{2}-\d{2}(。)",
    }
    text, curated_matches = re.subn(
        labels[lang], rf"\g<1>{updated}\g<2>", text
    )
    if curated_matches != 1:
        raise ValueError(
            f"expected one curated date in {lang} README, found {curated_matches}"
        )
    return text


def render_readme(data: dict[str, Any], lang: str, original: str) -> str:
    name = README_TARGETS[lang].name
    marker_errors = marker_layout_errors(original, name)
    if marker_errors:
        raise ValueError("; ".join(marker_errors))

    rendered = replace_counts(original, data)
    rendered = replace_updated(rendered, data, lang)
    rendered = replace_block(rendered, "TIMELINE", render_timeline(data, lang))
    rendered = replace_block(rendered, "PAPERS", render_papers(data, lang))
    rendered = replace_block(rendered, "REPOS", render_repos(data, lang))
    rendered = replace_block(rendered, "READINGS", render_readings(data, lang))
    rendered = rendered.rstrip() + "\n"

    post_errors = marker_layout_errors(rendered, name)
    if post_errors:
        raise ValueError("; ".join(post_errors))
    return rendered


def _write_temp(path: Path, content: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def atomic_write_all(
    updates: dict[Path, str], originals: dict[Path, str]
) -> None:
    """Prepare every file first, then atomically replace with best-effort rollback."""
    temp_paths: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for path, content in updates.items():
            temp_paths[path] = _write_temp(path, content)
        for path in updates:
            os.replace(temp_paths[path], path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            recovery = _write_temp(path, originals[path])
            os.replace(recovery, path)
        raise
    finally:
        for temp_path in temp_paths.values():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when generated sections are stale",
    )
    args = parser.parse_args()

    try:
        data = load_json(CATALOG_PATH)
        changelog = load_json(CHANGELOG_PATH)
        originals_by_lang = {
            lang: path.read_text(encoding="utf-8")
            for lang, path in README_TARGETS.items()
        }
    except (OSError, ValueError) as exc:
        print(f"Catalog render setup failed: {exc}", file=sys.stderr)
        return 1

    validation_errors = validate_catalog(
        data,
        changelog=changelog,
        readmes={
            README_TARGETS[lang].name: text
            for lang, text in originals_by_lang.items()
        },
    )
    if validation_errors:
        print("Refusing to render an invalid catalog:", file=sys.stderr)
        for error in validation_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    try:
        rendered_by_lang = {
            lang: render_readme(data, lang, original)
            for lang, original in originals_by_lang.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Catalog render failed before any file was written: {exc}", file=sys.stderr)
        return 1

    stale = [
        lang
        for lang in README_TARGETS
        if rendered_by_lang[lang] != originals_by_lang[lang]
    ]
    if stale and args.check:
        names = ", ".join(README_TARGETS[lang].name for lang in stale)
        print(f"Generated catalog sections are stale: {names}", file=sys.stderr)
        print("Run: make -C megakernel_landscape all", file=sys.stderr)
        return 1

    if stale:
        updates = {
            README_TARGETS[lang]: rendered_by_lang[lang] for lang in stale
        }
        originals = {
            README_TARGETS[lang]: originals_by_lang[lang] for lang in stale
        }
        try:
            atomic_write_all(updates, originals)
        except OSError as exc:
            print(f"Atomic README update failed: {exc}", file=sys.stderr)
            return 1
        for lang in stale:
            print(f"Updated {README_TARGETS[lang].name}")
    elif args.check:
        print("Generated catalog sections are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
