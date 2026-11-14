# Shadow skill microevaluation

Run `node cli/skill-microeval.mjs --config /private/pilot.json`. The JSON specifies
`directory` (new output directory), `skills.active` and `skills.candidate` (immutable
version directories), `profile` (the usual native Pi provider profile), `python`,
`runs`, elapsed `seconds` per sample and optional `totalSeconds` for the whole pilot. Set `taskModule` to the absolute file URI
of this directory's `tasks.mjs`, or provide explicit `tasks` with independent
Python `checker` paths. Reusing an output directory is refused rather than
refilling interrupted allocations.

The four frozen host-contract repairs cover stale compiled-launch arguments,
ragged store masks, a held-out strided layout contract, and recovery decisions.
The recovery task exercises runtime identity, harness-owned lease failures,
paired duration evidence, and validation/acceptance budget decisions. Its checker
records executed action callbacks and checks that the harness lease stays intact.
It injects global, preflight, optimisation, validation and acceptance guidance
alongside the implementation/debug/test scopes covered by the other tasks.
They use native Pi
read/write/edit/bash tools and behavioral Python oracles, without curriculum,
NPU allocation, campaign mutations or skill promotion. The first two tasks target
observed failure classes; strided layout is a holdout. These tasks measure short
engineering decisions, not device kernel correctness, throughput or generalization
across full optimization projects. Easy tasks can produce a ceiling effect.

`manifest.json` freezes task, checker and mapped skill contents. `results.json`
checkpoints each sample's correctness, elapsed time, provider tokens/cost and
native session path. Only selected task scopes are injected. Native Pi owns
provider retries, context and tool execution. Missing usage stays unknown, not
zero. Arm order alternates by repetition and samples run serially so this can
operate beside campaign work without competing for accelerator devices.

New campaign comparisons contain sixteen samples at two runs per arm, retaining
the 90-second per-sample ceiling and 900-second total elapsed allocation. An
incomplete comparison remains inconclusive. Existing comparisons retain their
original task IDs and frozen manifest when resumed after a catalogue upgrade.

Target execution cleanup retains a separate 30-second grace beyond a sample's
work allocation. It can cancel and observe already admitted work; it cannot admit
actor or verifier work after the original deadline. Late output never turns an
expired sample into a success. Cleanup time is charged to elapsed telemetry and
remaining study allocation, including across resume; it does not refill work.
