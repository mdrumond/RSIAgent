# Short campaign skill evaluations

Enable `skills.microEvaluation` for new comparisons to run the packaged CPU
contract tasks with native Pi. Defaults are two repetitions, 90 elapsed seconds
per sample, 900 elapsed seconds per comparison, and a 5% aggregate time/token
allowance. Candidate correctness must pass every check. This is host-contract
screening evidence, not NPU kernel-performance evidence.

The existing proposal, active-version publication guard and KDB/storage skill
recording remain responsible for candidate generation and promotion. Frozen
pending configuration chooses the evaluator; changing live configuration does
not convert a pending full-project round. Microtask artifacts remain plain and
are excluded from the full-instance KDB compression path.

After pausing and confirming the old campaign/evaluation executions stopped,
call `restartMicroEvaluation(await readConfig(root), {settings: {seconds: 90,
totalSeconds: 900, runs: 2, margin: 0.05}})` from `cli/micro-evaluation.mjs`.
It saves the original evolution metadata in `skills/restarts`, preserves previous
comparison directories and history, and starts a fresh comparison of the current
candidate against active skills. Project IDs, attempts, sessions and campaign
elapsed allocations are preserved. Future Phase 2 evidence uses the same micro
policy through the ordinary background pipeline.

Additional `allocationId`/`evaluationSeconds` grants are rejected for micro rounds;
use an explicit new comparison policy instead. Legacy full-project grants retain
their existing behavior.

A comparison reaching its elapsed deadline terminates inconclusively with retained
partial evidence; it does not block later evidence from creating another candidate.
Completed samples are retained across normal pauses. An interrupted native Pi
session resumes with its edited source and remaining sample/comparison elapsed
allocation. A crash leaving unknown in-flight execution remains fenced; a dead
controller alone is not proof that detached work stopped. Serialize resumable
runner calls using the campaign's existing skill lock.
