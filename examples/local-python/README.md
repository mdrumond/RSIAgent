# Local Python skill-evolution integration tests

Run the deterministic integration tests from the repository root with Node 22.23+ and
Python 3.11+ plus the normal host dependencies:

```bash
DRIVER_PYTHON=python3 node --test --test-concurrency=1 tests/ascend/local-python.mjs
```

Two small projects exercise tail reductions and signed-integer parsing. A local
scripted OpenAI-compatible server drives the real native Pi sessions, CLI waves,
Python subprocesses, independent frozen verifiers, and isolated A/B evaluator.
The scripted actor copies code from the implementation skill it actually reads;
no repair occurs in the baseline arm. The writer's candidate either repairs both
contracts or introduces a sign regression that the parser oracle rejects.

Integration coverage includes failure-only evidence, two accumulated proposals without any
benchmark evaluation, an explicit elapsed allocation, candidate correctness,
promotion withholding, regression rejection, and
interruption/resume of the same evaluation instance without rerunning completed
samples. Native sessions, provider streams, logs and verifier outputs are retained
when a test fails. Set `DRIVER_ARTIFACTS` to an existing directory to retain passing
evidence too. Tests execute serially and make no paid-provider or NPU requests;
copied evaluation machinery uses the normal dependency installer. The integration
configuration uses one sample per arm for each of the two controls: four project
executions per round. This checks scheduling, data propagation and promotion
mechanics with known scripted outcomes. A correct one-sample candidate remains
unpromoted because it lacks repeated evidence; the existing two-run tests cover
promotion. These integration tests do not estimate skill-improvement
efficacy or justify changing benchmark evaluation policy.

## Optional real-provider experiment

This experiment uses the same workloads with a real configured actor and writer.
It can spend API tokens. Keep the bootstrap/profile JSON private; supply `baseline`
and `skills` paths, `python`, `actor` (and optionally `skill_writer_actor`), and
`skillSettings.evaluationSeconds`. Credential references use the ordinary actor
profile contract; do not put credentials in this example. Paths resolve from the
calling directory. The output directory must not exist.

```bash
RSI_LOCAL_PYTHON_LIVE=1 node examples/local-python/run.mjs /private/bootstrap.json /private/new-output
```

The runner forces local CPU execution, concurrency one, and a two-original tuning
cadence. Inherited campaign plans, completion targets and resource masks are
ignored, and evaluation controls are pinned to the generated reduction/parser
projects. The private input profile is unchanged. The runner retains the instance
and a `result.json` status report. Actor time
defaults to 60 seconds per project; the evaluation allocation must be explicit.
Do not interpret an inconclusive evaluation as an improvement. This is a separate
provider experiment, not part of required automated integration tests.

Deterministic integration tests establish orchestration and executable guidance
propagation. A successful provider experiment establishes behavior on these two
Python projects only; neither establishes Ascend correctness or broad benchmark
improvement.
