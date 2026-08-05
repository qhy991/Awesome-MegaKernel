#!/usr/bin/env python3
"""Shared loading and validation helpers for the MegaKernel catalog."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
CHANGELOG_PATH = ROOT / "data" / "changelog.json"
README_PATHS = {
    "README.md": REPO_ROOT / "README.md",
    "README.zh-CN.md": REPO_ROOT / "README.zh-CN.md",
}
MARKDOWN_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.zh-CN.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "megakernel-survey.zh-CN.md",
    REPO_ROOT / "docs" / "case-studies" / "mixture-of-kittens.zh-CN.md",
    ROOT / "README.md",
    ROOT / "docs" / "taxonomy.md",
    REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
)

ENTRY_KINDS = ("papers", "repos", "readings")
MARKERS = ("TIMELINE", "PAPERS", "REPOS", "READINGS")
VALID_SCOPES = {"core", "foundation", "adjacent"}
DATE_RE = re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?$")
MONTH_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")
FULL_DATE_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
)
SEMVER_RE = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"ref", "source"}

COMMON_REQUIRED = {
    "id",
    "name",
    "url",
    "category",
    "scope",
    "summary_en",
    "summary_zh",
    "related",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def valid_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def canonical_url(value: str) -> str:
    """Normalize a URL enough to detect common catalog duplicates."""
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    if host in {"github.com", "doi.org", "www.doi.org"}:
        path = path.lower()
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in TRACKING_QUERY_KEYS
            and not key.lower().startswith(TRACKING_QUERY_PREFIXES)
        )
    )
    return urlunparse(("https", host, path, "", query, ""))


def marker_layout_errors(text: str, name: str) -> list[str]:
    """Require one ordered, non-overlapping block for each generated section."""
    errors: list[str] = []
    spans: list[tuple[str, int, int]] = []
    for marker in MARKERS:
        start_token = f"<!-- CATALOG:{marker}:START -->"
        end_token = f"<!-- CATALOG:{marker}:END -->"
        start_count = text.count(start_token)
        end_count = text.count(end_token)
        if start_count != 1:
            errors.append(
                f"{name}: expected one {start_token}, found {start_count}"
            )
        if end_count != 1:
            errors.append(f"{name}: expected one {end_token}, found {end_count}")
        if start_count == 1 and end_count == 1:
            start = text.index(start_token)
            end = text.index(end_token)
            if start >= end:
                errors.append(f"{name}: {marker} END must follow START")
            spans.append((marker, start, end + len(end_token)))

    if len(spans) == len(MARKERS):
        previous_end = -1
        for expected, (actual, start, end) in zip(MARKERS, spans):
            if actual != expected:
                errors.append(
                    f"{name}: generated blocks must follow {MARKERS!r}"
                )
                break
            if start < previous_end:
                errors.append(
                    f"{name}: generated blocks overlap or are out of order"
                )
                break
            previous_end = end
    return errors


def markdown_link_errors(text: str, name: str, base: Path) -> list[str]:
    errors: list[str] = []
    for target in MARKDOWN_LINK_RE.findall(text):
        target = target.strip().strip("<>")
        if (
            target.startswith(("https://", "http://", "mailto:", "#"))
            or not target
        ):
            continue
        relative = unquote(target.split("#", 1)[0])
        if relative and not (base / relative).exists():
            errors.append(
                f"{name}: local link target does not exist: {relative!r}"
            )
    return errors


def _safe_text_errors(value: str, label: str) -> list[str]:
    errors: list[str] = []
    if "<!-- CATALOG:" in value:
        errors.append(f"{label}: generated marker tokens are not allowed")
    if "\r" in value:
        errors.append(f"{label}: carriage returns are not allowed")
    if any(ord(character) < 32 and character not in {"\n", "\t"} for character in value):
        errors.append(f"{label}: control characters are not allowed")
    return errors


def _category_ids(
    categories_root: dict[str, Any], kind: str
) -> tuple[set[str], list[str]]:
    errors: list[str] = []
    raw = categories_root.get(kind)
    if not isinstance(raw, list):
        errors.append(f"categories:{kind} must be a list")
        return set(), errors

    ids: list[str] = []
    for index, item in enumerate(raw):
        label = f"categories:{kind}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label}: must be an object")
            continue
        category_id = item.get("id")
        if not isinstance(category_id, str) or not ID_RE.fullmatch(category_id):
            errors.append(f"{label}: id must match {ID_RE.pattern}")
        else:
            ids.append(category_id)
        for field in ("id", "title_en", "title_zh", "note_en", "note_zh"):
            value = item.get(field)
            if not nonempty(value):
                errors.append(f"{label}: {field} must be a non-empty string")
            elif isinstance(value, str):
                errors.extend(_safe_text_errors(value, f"{label}:{field}"))

    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"categories:{kind}: duplicate IDs {duplicates}")
    return set(ids), errors


def _date_is_future(value: str, today: date) -> bool:
    if len(value) == 4:
        return int(value) > today.year
    return value > today.strftime("%Y-%m")


def _validate_changelog(
    changelog: Any, data: dict[str, Any], today: date
) -> list[str]:
    errors: list[str] = []
    if not isinstance(changelog, list) or not changelog:
        return ["data/changelog.json: must be a non-empty list"]

    versions: list[str] = []
    dates: list[str] = []
    for index, item in enumerate(changelog):
        label = f"data/changelog.json[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label}: must be an object")
            continue
        version = item.get("version")
        change_date = item.get("date")
        summary = item.get("summary")
        if not isinstance(version, str) or not version.startswith("v") or not SEMVER_RE.fullmatch(version[1:]):
            errors.append(f"{label}: version must be v-prefixed semantic version")
        else:
            versions.append(version)
        if not isinstance(change_date, str) or not FULL_DATE_RE.fullmatch(change_date):
            errors.append(f"{label}: date must be YYYY-MM-DD")
        else:
            try:
                parsed_date = date.fromisoformat(change_date)
            except ValueError:
                errors.append(f"{label}: date is not a real calendar date")
            else:
                dates.append(change_date)
                if parsed_date > today:
                    errors.append(f"{label}: date {change_date} is in the future")
        if not nonempty(summary):
            errors.append(f"{label}: summary must be a non-empty string")
        elif isinstance(summary, str):
            errors.extend(_safe_text_errors(summary, f"{label}:summary"))

    duplicate_versions = sorted(
        key for key, count in Counter(versions).items() if count > 1
    )
    if duplicate_versions:
        errors.append(
            f"data/changelog.json: duplicate versions {duplicate_versions}"
        )
    if dates != sorted(dates, reverse=True):
        errors.append("data/changelog.json: entries must be newest first")

    latest = changelog[0]
    if isinstance(latest, dict):
        if latest.get("version") != f"v{data.get('version')}":
            errors.append(
                "data/changelog.json: latest version does not match catalog"
            )
        if latest.get("date") != data.get("updated"):
            errors.append("data/changelog.json: latest date does not match catalog")
    return errors


def validate_catalog(
    data: Any,
    *,
    changelog: Any | None = None,
    readmes: dict[str, str] | None = None,
    markdown_docs: Iterable[tuple[str, str, Path]] = (),
    today: date | None = None,
) -> list[str]:
    """Return all validation errors without raising on malformed input."""
    errors: list[str] = []
    today = today or date.today()
    if not isinstance(data, dict):
        return ["data/catalog.json: top level must be an object"]

    for key in ("version", "updated", "categories", *ENTRY_KINDS):
        if key not in data:
            errors.append(f"top level: missing {key!r}")

    version = data.get("version")
    if not isinstance(version, str) or not SEMVER_RE.fullmatch(version):
        errors.append("top level: version must be a semantic version")

    updated = data.get("updated")
    if not isinstance(updated, str) or not FULL_DATE_RE.fullmatch(updated):
        errors.append("top level: updated must be YYYY-MM-DD")
    else:
        try:
            updated_date = date.fromisoformat(updated)
        except ValueError:
            errors.append("top level: updated is not a real calendar date")
        else:
            if updated_date > today:
                errors.append(f"top level: updated {updated} is in the future")

    categories_value = data.get("categories")
    if not isinstance(categories_value, dict):
        errors.append("top level: categories must be an object")
        categories_root: dict[str, Any] = {}
    else:
        categories_root = categories_value

    category_sets: dict[str, set[str]] = {}
    for kind in ENTRY_KINDS:
        category_sets[kind], category_errors = _category_ids(
            categories_root, kind
        )
        errors.extend(category_errors)

    all_entries: list[tuple[str, int, dict[str, Any]]] = []
    for kind in ENTRY_KINDS:
        entries = data.get(kind)
        if not isinstance(entries, list):
            errors.append(f"{kind}: must be a list")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{kind}[{index}]: must be an object")
                continue
            all_entries.append((kind, index, entry))

    string_ids = [
        entry["id"]
        for _, _, entry in all_entries
        if isinstance(entry.get("id"), str)
    ]
    for duplicate, count in Counter(string_ids).items():
        if count > 1:
            errors.append(f"duplicate id {duplicate!r} appears {count} times")
    known_ids = set(string_ids)

    primary_urls: dict[str, list[str]] = {}
    for kind, index, entry in all_entries:
        raw_id = entry.get("id")
        display_id = raw_id if isinstance(raw_id, str) else index
        label = f"{kind}:{display_id}"
        missing = sorted(COMMON_REQUIRED - set(entry))
        if missing:
            errors.append(f"{label}: missing fields {missing}")

        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not ID_RE.fullmatch(entry_id):
            errors.append(f"{label}: id must match {ID_RE.pattern}")

        name = entry.get("name")
        if not nonempty(name):
            errors.append(f"{label}: name must be a non-empty string")
        elif isinstance(name, str):
            errors.extend(_safe_text_errors(name, f"{label}:name"))

        url = entry.get("url")
        if not valid_url(url):
            errors.append(f"{label}: url must be an absolute HTTPS URL")
        elif isinstance(url, str):
            primary_urls.setdefault(canonical_url(url), []).append(label)

        entry_date = entry.get("date")
        if kind == "papers" and entry_date is None:
            errors.append(f"{label}: missing field 'date'")
        if entry_date is not None:
            if not isinstance(entry_date, str) or not DATE_RE.fullmatch(entry_date):
                errors.append(f"{label}: date must be YYYY or YYYY-MM when set")
            elif _date_is_future(entry_date, today):
                errors.append(f"{label}: date {entry_date} is in the future")

        scope = entry.get("scope")
        if not isinstance(scope, str) or scope not in VALID_SCOPES:
            errors.append(f"{label}: invalid scope {scope!r}")
        category = entry.get("category")
        if not isinstance(category, str) or category not in category_sets[kind]:
            errors.append(f"{label}: invalid category {category!r}")

        for field in ("summary_en", "summary_zh"):
            value = entry.get(field)
            if not nonempty(value):
                errors.append(f"{label}: {field} must be a non-empty string")
            elif isinstance(value, str):
                errors.extend(_safe_text_errors(value, f"{label}:{field}"))

        related = entry.get("related")
        if not isinstance(related, list):
            errors.append(f"{label}: related must be a list")
        else:
            related_strings: list[str] = []
            for related_id in related:
                if not isinstance(related_id, str):
                    errors.append(
                        f"{label}: related IDs must be non-empty strings"
                    )
                    continue
                related_strings.append(related_id)
                if related_id not in known_ids:
                    errors.append(f"{label}: unknown related id {related_id!r}")
                if isinstance(entry_id, str) and related_id == entry_id:
                    errors.append(f"{label}: cannot relate to itself")
            repeated = sorted(
                key
                for key, count in Counter(related_strings).items()
                if count > 1
            )
            if repeated:
                errors.append(f"{label}: duplicate related IDs {repeated}")

        for optional_url in ("code", "docs", "project"):
            value = entry.get(optional_url)
            if value is not None and not valid_url(value):
                errors.append(f"{label}: {optional_url} must be null or HTTPS URL")

        if kind == "papers":
            venue = entry.get("venue")
            if not nonempty(venue):
                errors.append(f"{label}: venue must be a non-empty string")
            elif isinstance(venue, str):
                errors.extend(_safe_text_errors(venue, f"{label}:venue"))
            topics = entry.get("topics")
            if not isinstance(topics, list) or not topics:
                errors.append(f"{label}: topics must be a non-empty list")
            elif not all(nonempty(tag) for tag in topics):
                errors.append(f"{label}: topics must contain non-empty strings")
            else:
                topic_strings = [str(tag) for tag in topics]
                if len(topic_strings) != len(set(topic_strings)):
                    errors.append(f"{label}: topics must not contain duplicates")
                for tag in topic_strings:
                    errors.extend(_safe_text_errors(tag, f"{label}:topics"))
                    if "`" in tag:
                        errors.append(f"{label}: topic tags cannot contain backticks")
        elif kind == "repos":
            for field in ("target_en", "target_zh", "status_en", "status_zh"):
                value = entry.get(field)
                if not nonempty(value):
                    errors.append(f"{label}: {field} must be a non-empty string")
                elif isinstance(value, str):
                    errors.extend(_safe_text_errors(value, f"{label}:{field}"))
        elif kind == "readings":
            for field in ("kind_en", "kind_zh"):
                value = entry.get(field)
                if not nonempty(value):
                    errors.append(f"{label}: {field} must be a non-empty string")
                elif isinstance(value, str):
                    errors.extend(_safe_text_errors(value, f"{label}:{field}"))

        timeline_present = any(
            key in entry for key in ("timeline_label", "timeline_label_en", "timeline_label_zh")
        )
        if "timeline_label" in entry:
            errors.append(
                f"{label}: use timeline_label_en and timeline_label_zh"
            )
        if timeline_present:
            for field in ("timeline_label_en", "timeline_label_zh"):
                value = entry.get(field)
                if not nonempty(value):
                    errors.append(f"{label}: {field} must be a non-empty string")
                elif isinstance(value, str):
                    errors.extend(_safe_text_errors(value, f"{label}:{field}"))
                    if "\n" in value or ":" in value:
                        errors.append(
                            f"{label}: {field} cannot contain newlines or colons"
                        )
            if not isinstance(entry_date, str) or not MONTH_RE.fullmatch(entry_date):
                errors.append(f"{label}: timeline labels require a YYYY-MM date")

    for canonical, labels in primary_urls.items():
        if len(labels) > 1:
            errors.append(
                f"duplicate primary URL {canonical!r}: {', '.join(labels)}"
            )

    for kind in ENTRY_KINDS:
        raw_categories = categories_root.get(kind)
        if not isinstance(raw_categories, list):
            continue
        used = {
            entry.get("category")
            for entry_kind, _, entry in all_entries
            if entry_kind == kind and isinstance(entry.get("category"), str)
        }
        unused = sorted(category_sets[kind] - used)
        if unused:
            errors.append(f"categories:{kind}: unused IDs {unused}")

    if changelog is not None:
        errors.extend(_validate_changelog(changelog, data, today))

    if readmes is not None:
        for name, text in readmes.items():
            errors.extend(marker_layout_errors(text, name))

    for name, text, base in markdown_docs:
        errors.extend(markdown_link_errors(text, name, base))

    return errors
