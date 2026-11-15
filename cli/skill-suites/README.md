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
