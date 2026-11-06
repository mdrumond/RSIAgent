# Executable Phase 2 project contracts

`ascend_tools.baselines.projects.prepare_project(workload, target, root, receipt,
identity, python='python3', project_id=None)` validates a frozen qualification
receipt and creates a new independent project. Targets are `a3` (Triton-Ascend)
and `a5` (Catlass DSL). Machine interpreter/runtime settings are caller-supplied;
no remote transport, device selection or campaign launch occurs here.

The returned spec has `source`, `goal`, `actor_seconds`, `phase`, `performance_goal`
and `verifier.files/command/performance_command`. Pass it to the native project
importer and assign the current campaign entry. `source/kernel.py` initially
raises NotImplementedError. Actors implement `run(*args, **attrs)` using the
frozen PyTorch argument order and must return the complete tensor/state output
structure on the NPU, with declared dtypes and unchanged inputs. They may inspect
previous projects and choose a starting source, but copies belong to this project.
The reference `Model.forward`/golden signature and every case are inside the
frozen archive; GDN's attributes are positional, while Level 5 attrs are keyword
arguments. None final states remain None when the contract disables them.

Only two protected files need generic CPL transport: `frozen/verify.py` and
`frozen/verifier.tar.gz`. The archive contains the trusted package, baseline
assets, qualification receipt and runtime identity. The launcher verifies its
archive hash, extracts by archive identity into a private cache, and validates
cached file bytes before reuse. The package is independent of shared/editable
source checkouts. Native import preserves relative paths and verifies protected
file hashes. Generated archives, identities and project roots remain ignored
runtime artifacts, not source-controlled machine configuration.

Correctness executes every frozen case with independent CPU-reference results,
all output tensors/states, NPU placement, dtype, finiteness and mutation checks.
A reduced case subset never passes the full contract. The production CLI has no
CPU fallback or case-subset option. Local tests call the same numerical core on
CPU and explicitly report `targetEvidence: false`; they are not NPU evidence.
Per-case progress and numerical input/output hashes survive ordinary failures.

Performance rechecks correctness and measures the freshly qualified PyTorch
reference and candidate on the same inputs/device, after five warmups and twenty
alternating paired repetitions. Input construction and numerical comparisons
are outside timing. Median-per-case speedups are combined by equal-case geometric
mean; all numerical cases and geometric speedup strictly greater than one are
required. Raw timings are retained; upstream metadata is not substituted for
measurement. This reports synchronized call-wall latency, not profiler-derived
kernel duration. The caller must provide healthy idle isolation.

Runtime versions/device identity and frozen hashes must match qualification.
Candidate source identity is retained and changes during verification fail.
Target language is part of the immutable contract and actor goal, but this
adapter does not attest compiler/backend language from NPU output placement.
`languageAttested` remains false. Preserve ordinary source/build logs/compiler
artifacts as evidence; do not reinterpret that flag as backend attestation.
