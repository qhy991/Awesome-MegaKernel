"""Regression tests for catalog validation and rendering."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import unittest

from catalog_lib import (
    CATALOG_PATH,
    CHANGELOG_PATH,
    README_PATHS,
    load_json,
    marker_layout_errors,
    validate_catalog,
)
from render_catalog import escape_markdown_label, render_readme


class CatalogValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_json(CATALOG_PATH)
        cls.changelog = load_json(CHANGELOG_PATH)
        cls.readmes = {
            name: path.read_text(encoding="utf-8")
            for name, path in README_PATHS.items()
        }

    def validate(self, data: object) -> list[str]:
        return validate_catalog(
            data,
            changelog=self.changelog,
            readmes=self.readmes,
            today=date(2026, 7, 28),
        )

    def test_current_catalog_is_valid(self) -> None:
        self.assertEqual(self.validate(self.data), [])

    def test_bad_categories_type_is_reported_without_traceback(self) -> None:
        mutated = deepcopy(self.data)
        mutated["categories"] = []
        errors = self.validate(mutated)
        self.assertTrue(any("categories must be an object" in item for item in errors))

    def test_unhashable_id_is_reported_without_traceback(self) -> None:
        mutated = deepcopy(self.data)
        mutated["papers"][0]["id"] = {}
        errors = self.validate(mutated)
        self.assertTrue(any("id must match" in item for item in errors))

    def test_reversed_marker_is_rejected(self) -> None:
        original = self.readmes["README.md"]
        start = "<!-- CATALOG:PAPERS:START -->"
        end = "<!-- CATALOG:PAPERS:END -->"
        mutated = original.replace(start, "__START__", 1)
        mutated = mutated.replace(end, start, 1).replace("__START__", end, 1)
        errors = marker_layout_errors(mutated, "README.md")
        self.assertTrue(any("END must follow START" in item for item in errors))

    def test_marker_in_catalog_text_is_rejected(self) -> None:
        mutated = deepcopy(self.data)
        mutated["papers"][0]["name"] = "<!-- CATALOG:PAPERS:END -->"
        errors = self.validate(mutated)
        self.assertTrue(any("marker tokens" in item for item in errors))

    def test_markdown_link_labels_escape_table_delimiters(self) -> None:
        self.assertEqual(escape_markdown_label(r"A\B | [C]"), r"A\\B \| \[C\]")

    def test_updated_date_participates_in_rendering(self) -> None:
        mutated = deepcopy(self.data)
        mutated["updated"] = "2026-07-27"
        rendered = render_readme(mutated, "en", self.readmes["README.md"])
        self.assertIn("Last%20Updated-2026--07--27", rendered)
        self.assertIn("**Last curated:** 2026-07-27.", rendered)

    def test_chinese_generated_labels_are_localized(self) -> None:
        rendered = render_readme(
            self.data, "zh", self.readmes["README.zh-CN.md"]
        )
        self.assertIn("**工程博客。**", rendered)
        self.assertNotIn("**Engineering blog.**", rendered)

    def test_render_fails_before_mutating_input_on_missing_marker(self) -> None:
        original = self.readmes["README.zh-CN.md"]
        malformed = original.replace("<!-- CATALOG:READINGS:END -->", "", 1)
        with self.assertRaises(ValueError):
            render_readme(self.data, "zh", malformed)
        self.assertEqual(
            malformed.count("<!-- CATALOG:READINGS:END -->"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
