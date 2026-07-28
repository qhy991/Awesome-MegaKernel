# Contributing

Thanks for helping grow **Awesome MegaKernel**. Keep pull requests focused,
evidence-based, and easy to review.

## Inclusion boundaries

Classify each proposed entry in the pull request.

### Core

The work directly designs, generates, compiles, runs, or evaluates a
megakernel, mega-kernel, or multi-operator persistent kernel for LLM inference
or training. Examples include:

- end-to-end or multi-operator persistent LLM kernels;
- megakernel compilers, runtimes, schedulers, and agent synthesizers;
- MoE or multi-GPU megakernels with fine-grained compute/communication overlap;
- serving systems with a documented megakernel implementation path.

### Adjacent

The work is not itself an LLM megakernel, but offers direct, technically
specific evidence that helps understand or build one. Examples include
megakernel-versus-wavefront studies in another domain, a relevant persistent
runtime, or a serving-system case study.

State the connection to LLM megakernels in the pull request. Place the entry in
an explicitly adjacent section; do not present it as core work.

### Foundation

The work establishes a reusable mechanism, definition, or historical lineage
for the field, such as persistent threads, grid-wide synchronization, task
graphs, software pipelining, or an authoritative implementation tutorial.

Foundation entries must be primary or authoritative and must explain their
direct bridge to modern megakernels. Generic CUDA introductions and unrelated
single-operator optimization guides are out of scope.

### Usually out of scope

- single-operator kernels unless a core work explicitly uses them as a building
  block;
- generic compiler, serving, or AI-agent work with no megakernel mechanism;
- pure framework wrappers with no fused or persistent kernel implementation;
- closed-source marketing pages when no primary technical artifact is
  available;
- abandoned forks, mirrors, duplicate versions, and link aggregators.

## Evidence and links

Every entry must include at least one primary or authoritative link:

- papers: publisher, conference, arXiv, OpenReview, or DOI page;
- code: the canonical repository owned by the authors or organization;
- blogs or tutorials: the original author, laboratory, vendor, or project site;
- documentation: the official project or platform documentation.

Secondary summaries may be included as additional reading, but not as the only
evidence. Do not link to search results, scraped copies, tracking URLs, or an
unofficial mirror when the primary artifact is available.

Verify titles, dates, venue, authors, code ownership, and topic tags against the
primary source. Do not infer author roles or claim that a repository implements
a paper unless the primary source establishes that relationship.

## Performance claims

Only include a numerical performance claim when its evidence link provides
enough context to interpret it. Record, in the table or accompanying text:

- hardware: accelerator model, count, and relevant interconnect/topology;
- workload: model/operator, phase, precision, batch size, and sequence/context
  length when applicable;
- software: framework, compiler/runtime, and relevant version or commit;
- baseline: exact system, version, configuration, and comparison metric;
- date: paper/release date or the date on which the benchmark was measured;
- evidence: a direct paper table/figure/section, official report, or
  reproducible benchmark result.

Preserve qualifiers such as peak, median, end-to-end, kernel-only, or simulated.
If the required context cannot be verified, omit the number and describe the
qualitative contribution instead.

## Adding or updating an entry

1. Search `megakernel_landscape/data/catalog.json` for the exact title, project
   name, arXiv/DOI identifier, canonical repository slug, and aliases.
2. Read `megakernel_landscape/docs/taxonomy.md`, then add the entry to the
   appropriate catalog array, scope, and topic category.
3. Use concise, neutral descriptions and provide both `summary_en` and
   `summary_zh` (plus both language variants of repository target/status,
   reading kind, and timeline-label fields).
4. Connect related papers, repositories, and readings through existing catalog
   IDs; do not duplicate an artifact to express a relationship.
5. Bump `version` and `updated`, then prepend the same version/date to
   `megakernel_landscape/data/changelog.json`.
6. Run `make -C megakernel_landscape all` to regenerate both READMEs, count
   badges, curated dates, and the timeline.
7. Keep one pull request to one coherent theme, such as a paper and its official
   code.

`catalog.json` is the source of truth. Do not hand-edit content between
`CATALOG:*` markers in either README. The renderer keeps the generated catalog
blocks, count badges, and curated dates synchronized while using
language-specific descriptive fields. Hand-written introductions and guidance
outside those regions still require an explicit bilingual review.

## Link, duplicate, and formatting checks

Before opening the pull request:

- open every new or changed URL and confirm that it resolves to the intended
  artifact without requiring an unrelated redirect;
- search for alternate spellings, prior preprint titles, arXiv/DOI IDs, and
  repository renames to avoid duplicates;
- use one canonical entry when a paper, project page, and repository describe
  the same work, with the other artifacts linked from that entry;
- confirm the generated chronological order and table-column alignment;
- run `make -C megakernel_landscape check`;
- run `git diff --check`;
- review the final diff for unrelated reformatting or accidental removal of
  existing entries.

Linked projects and papers retain their own licenses. Contributions to this
curated repository are made available under [CC0 1.0 Universal](LICENSE).
