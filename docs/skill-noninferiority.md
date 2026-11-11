# Observed non-inferiority for short skill evaluations

`evaluateNonInferiority` in `controller/skill-noninferiority.mjs` compares a
frozen set of paired baseline/candidate results. It returns eligibility and
diagnostics; it does not publish skills or change an existing evaluation round.

```js
const verdict = evaluateNonInferiority(results, {
  projectIds: ['layout', 'softmax'], runs: 2, margin: 0.05,
  metrics: ['elapsedSeconds', 'tokens'],
});
```

Each observation contains `projectId`, zero-based `run`, `arm` (`active` or
`candidate`), boolean `correct`, and measured positive `elapsedSeconds` and
`tokens`. The task list, repetitions, metrics and margin must be selected before
running either arm. Missing observations, duplicate pair identities or unknown
correctness make the comparison inconclusive. Every candidate correctness check
must pass, and at least two repetitions are required.

For each requested resource metric, compute candidate/baseline cost ratios within
each repetition, their geometric mean within each task, then the geometric mean
across tasks. Thus each task has equal weight despite differences in scale.
Every selected aggregate metric must be at most `1 + margin`; individual pairs
may lose. Per-task and per-pair ratios remain in the report so a localized
regression is visible. Improvements in time cannot conceal a token regression.

Token counts mean observed agent resource use; elapsed time means observed
evaluation cost. Neither measures device kernel latency. Callers may explicitly
select only time when token accounting is unavailable. Missing, nonfinite or
nonpositive costs otherwise remain inconclusive and are never synthesized as
zero. The gate does not infer elapsed costs from legacy signed `score` fields.

This is an observed comparison on the selected microtasks, not a statistical
confidence bound or proof of non-inferiority on full optimization projects.
Shadow callers should retain the frozen inputs and complete report before
deciding whether to enable promotion.
