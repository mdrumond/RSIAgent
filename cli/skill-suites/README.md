# Curated skill evaluation packs

`a3-packs.json` imports the retained A3 skill-objective fixtures, candidate starters,
known-good implementations and protected numerical specifications. It contains
five families with five public cases each, private seed variants and contract
edges. These are source templates, not a declaration of current device validity.

The loader embeds only trusted checker dependencies into each frozen checker.
Actors receive a candidate starter, workload, public case and public checker;
they edit `kernel.py`. Correctness packs retain the original broken starter as a
negative. Optimization packs use a deliberately corrupted independent control;
a slower but correct starter is not a correctness negative.

Before admission, the existing target environment supplies a fresh runtime
identity and runs correct-reference and incorrect-control qualification. The
identity covers the actual product, package versions/records/source hashes,
compiler hashes and CANN version-file hashes. Qualified controls are reused only
for the same pack and semantic runtime identity. They do not establish actor
success or kernel speedup: those require the existing frozen A/B experiment.

CPU numeric tests verify comparator behavior on bounded existing fixtures only.
They do not establish NPU arithmetic, compilation, first-call launch or timing.
The packs are A3 templates; they do not establish A5 compatibility.

Case discovery and semantic selection use local catalogue descriptors before
requesting a device. Only a selected curated A3 pack requests this pack's
Triton-based runtime identity; CPU diagnostics and existing A5 catalogues retain
their target-specific paths. Configured catalogues may scope applicability with
`proposalIds`, `backlogIds` or `scopes`; final semantic selection still applies.

`qualifyHypothesisTasks` retains its public signature and root directory. It
reuses an already-qualified root without executing controls. Known transient
harness/closed-transport failures may continue only after positive closure,
under the first attempt's stored elapsed allocation. The original blocked report
stays immutable; `qualification-continuation.json` indexes the current child
attempt under `continuations/`. Root settlement follows that index before any
child can be considered closed, and the returned `directory` identifies the
actual report. Passed control receipts and cumulative elapsed time carry forward;
bad controls, invalid receipts, elapsed failures and unresolved work do not become
qualified. A larger allowance from a later caller cannot renew the first one.
