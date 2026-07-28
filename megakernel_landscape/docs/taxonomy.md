# Taxonomy and inclusion policy

The catalog separates **scope** from **topic**. Scope answers “how directly is
this a MegaKernel work?” Topic answers “which part of the landscape does it
cover?”

## Scope

### `core`

The artifact directly designs, generates, compiles, runs, or evaluates an
end-to-end or multi-operator persistent kernel for LLM inference or training.
MoE and distributed kernels qualify when computation and communication are
scheduled together inside the persistent kernel.

### `foundation`

The artifact contributes a mechanism reused by modern MegaKernel systems:
persistent threads, tile/task graphs, software pipelining, fine-grained
synchronization, GPU-initiated communication, or a directly used kernel DSL.
It must have an explicit bridge to a core work.

### `adjacent`

The artifact studies a close alternative, a smaller fusion scope, a
non-LLM analogue, or a serving system that uses persistent kernels without
forming a whole-model MegaKernel. Its relationship must be explained.

## Topic categories

Papers and reports:

- `whole_model`: whole-model kernels, compilers, and persistent runtimes.
- `moe_distributed`: MoE, tensor/expert parallelism, and communication.
- `foundations`: historical and architectural foundations.
- `adjacent`: deep fusion, alternative execution models, and non-LLM studies.

Systems and code:

- `compiler_runtime`: compilers, whole-model artifacts, and runtimes.
- `model_specific`: hand-tuned or model/hardware-specific implementations.
- `distributed_moe`: distributed, communication, and MoE kernels.
- `building_blocks`: directly relevant DSLs, primitives, and runtimes.

## Evidence rules

- Prefer conference/publisher, arXiv, OpenReview, official documentation, and
  canonical author or organization repositories.
- Do not infer author roles, venue status, or paper-to-code ownership.
- Describe partial or binary-only releases explicitly.
- Numerical performance claims belong in the catalog only when hardware,
  workload, baseline, software configuration, metric, date, and primary
  evidence are all available.
- A project name containing “kernel” or “persistent” is not enough for
  inclusion.
