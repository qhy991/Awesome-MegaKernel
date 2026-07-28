# MegaKernel Landscape Maintainer

This zero-dependency toolkit keeps the English and Chinese Awesome MegaKernel
catalogs synchronized.

```text
data/catalog.json
      ├── validate_catalog.py
      └── render_catalog.py ──> ../README.md
                              └> ../README.zh-CN.md
```

`data/catalog.json` is the source of truth for papers, systems, repositories,
and readings. The renderer only changes content between `CATALOG:*` markers,
the three count badges, and the two curated-date fields in the root READMEs.
It prepares and validates both language versions before replacing either file.

## Commands

```bash
make validate   # schema, IDs, categories, dates, URLs, translations, relations
make render     # validate, then regenerate both READMEs transactionally
make test       # exercise malformed-data and marker-corruption regressions
make check      # validate, test, and fail if generated sections are stale
make all        # validated render
```

The scripts use only the Python standard library. No virtual environment or
package installation is required.

## Add an entry

1. Read [`docs/taxonomy.md`](docs/taxonomy.md).
2. Add the artifact to the appropriate array in `data/catalog.json`.
3. Provide both `_en` and `_zh` descriptive fields.
4. Link the primary paper page, canonical repository, or original article.
5. Use `related` IDs to connect papers, repositories, and readings.
6. Bump the semantic `version` and `updated` date, then prepend the matching
   entry to `data/changelog.json`.
7. Run `make all` and review both root READMEs. Dated entries sort newest first;
   undated entries sort alphabetically after them.

Do not edit generated catalog blocks by hand.
