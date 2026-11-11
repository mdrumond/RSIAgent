# Shadow skill microevaluation

Run `node cli/skill-microeval.mjs --config /private/pilot.json`. The JSON specifies
`directory` (new output directory), `skills.active` and `skills.candidate` (immutable
version directories), `profile` (the usual native Pi provider profile), `python`,
`runs`, elapsed `seconds` per sample and optional `totalSeconds` for the whole pilot. Set `taskModule` to the absolute file URI
of this directory's `tasks.mjs`, or provide explicit `tasks` with independent
Python `checker` paths. Reusing an output directory is refused rather than
refilling interrupted allocations.

The three frozen host-contract repairs cover stale compiled-launch arguments,
ragged store masks, and a held-out strided layout contract. They use native Pi
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
