# Fresh hypothesis-specific skill comparisons

`prepareHypothesisTasks(directory)` from `cli/microtasks/hypotheses.mjs` produces
15 frozen CPU tasks, a `byHypothesis` task map and `catalogue.json` metadata.
Use a pinned source checkout before preparing the catalogue. The manifest retains
prompt/input/checker hashes and the production admission/compiler/oracle hashes.
Pass the selected five tasks directly to the existing `runMicroeval`; use the
unchanged noninferiority gate with two repetitions per arm and the existing 5%
aggregate elapsed/token margin. Five distinct cases are not five repetitions.

The distinct hypotheses are:

- `fault-ownership`: lease closure, runtime identity, candidate traceback,
  protected traceback and missing-origin evidence. The two saved A3 fault-reporting
  proposals render identical advice; compare that advice once while retaining both
  proposal IDs/provenances in the experiment record.
- `construction-admissibility`: directory-valued fixtures, verifier ownership,
  CPU-command nesting, baseline entrypoint identity and implemented command modes.
  The checker calls the real pinned production schema/static admission helper.
  In particular, `baseline_construction_command` is a **top-level** proposal field;
  the original saved guidance claiming nested-only placement is not corrected for
  this comparison. Invalid modes require an actual CPU command check, not just
  static admission. Goals and target command contracts cannot be removed to pass.
- `dtype-triage`: unsupported mutable-source dtype, warning-only output, a failed
  frozen output contract, valid CPU reference precision and an unsupported frozen
  requirement. Supplied capabilities apply only to the named synthetic operator
  and binding. These are CPU reasoning cases, not assertions about Torch-NPU,
  Catlass DSL or actual Ascend dtype support. Proof freshness and supported-double
  variations prevent inferring validity or a blanket ban from one warning.

The v2 catalogue gives every ownership/dtype case an answer-changing held-out
condition as well as path/binding variations. Constant public-answer negative
controls must fail every such case. Construction repair arguments must remain
unmodified, even when the returned proposal is admissible. Each case has a public
incident and two held-out variants. Checkers test behavior,
independent admission and immutable fixture preservation; they do not score prose
or match proposed skill wording. Positive and negative controls are in
`tests/ascend/skill-hypotheses.mjs`; production oracle controls are in
`tests/ascend/construction-oracle.mjs`. Finish those before model comparisons and
freeze original candidate advice unchanged. A learned revision is a separate arm
and comparison, not a replacement for an original failed result.

For question projects, the trusted oracle is available as:

```
node cli/microtasks/construction-oracle.mjs --root PROJECT --proposal proposal.json
node cli/microtasks/construction-oracle.mjs --root PROJECT --proposal proposal.json --controls
```

The first command observes schema/static admission only; `--controls` also checks
reference/negative/starter and CPU baseline behavior using native Pi execution.
`--control baseline` selects only that actual CPU control; unknown or unavailable
selectors are rejected rather than reporting an unchecked success. It reports its scope in
`checks`; it does not establish all other controls or device qualification.
No schema is changed to agree with a proposed lesson.

`recoveryDecisionsV2` is an opt-in new task ID clarifying that timestamp values,
not the positions of rows in an event list, determine valid start/end pairs.
The legacy task catalogue, checker, defaults and prior frozen experiments remain
unchanged and resume through their existing paths.
