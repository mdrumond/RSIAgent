# Isolated skill-project A/B

`runSkillProjectAB` in `cli/skill-project-ab.mjs` evaluates one trusted prepared
project under baseline and candidate guidance. It does not alter a campaign,
publish a skill, or replace the existing campaign evaluator. The caller supplies
the registered environment binding and scheduler resource; the existing Ascend
SDK owns health, transport, leases and execution settlement.

```js
const result = await runSkillProjectAB({
  directory, config, project, guidance: { baseline, candidate },
  evidence: [{ reference, text }], binding, resource,
  seconds: 1200, actorSeconds: 180, signal,
});
```

The directory must be new. Existing experiments are never restarted by this API,
including after interruption or unknown execution status. Their frozen inputs,
model sessions, raw receipts and pending custody remain available for inspection.
All stages share the supplied elapsed allocation. Actor allocations are equal;
insufficient remaining time stops admission rather than giving a later actor a
smaller allocation. Thirty seconds within that allocation are reserved for closure.

The prepared project contains:

- `id`, `objective: 'correctness'`, `target: 'a3' | 'a5'`, and a public `prompt`.
- A UTF-8 `files` mapping containing `kernel.py` and any public contract files.
- Exactly five `cases: [{ id, parameters }]`, with distinct IDs and parameter
  sets. Changing property order does not create a distinct parameter set.
- A trusted `checker` path. Its contents are frozen before any model runs.
- `controls.reference.files` and one or more `controls.negatives[].files`, each
  containing `kernel.py`. Control sources are never given to the A/B actors.

The trusted checker takes the proposal directory as its only argument. Its last
stdout line is a JSON receipt containing `correct`, `kernelSha256` and `evidence`.
A full pass requires exit 0, `correct: true`, an NPU `evidence.device` and all five
public IDs in `evidence.checkedCases`; additional seed variants are allowed.
A definitive source failure requires exit 1, `correct: false` and a nonempty
`evidence.reason`. Missing receipts, mismatched source/checker hashes, incomplete
coverage and infrastructure refusals remain unscored. Reference/negative controls
must be empirically discriminating. A historical negative that now passes is an
uninformative result requiring learning, never a reason to manufacture a failure.

The test designer and independent reviewer run in separate native Pi sessions.
They read `input.json` and write `proposal.json` / `qa.json` using stock file tools;
final prose is not the submission contract. Plans preserve the trusted cases and
explicitly state success, disconfirmation and inconclusive outcomes. Exact evidence
references and the fixed scoring policy are validated before admission. The review
cannot change the trusted oracle, code, cases or treatment policy.

After qualification, four independent proposal-only actor sessions run serially:
baseline/candidate, then candidate/baseline. Each gets the same starter, cases,
contract and elapsed allocation, differing only in guidance. Each edits only
`kernel.py`; trusted external verification then checks the whole project. Five
cases are correlated observations of a proposal, not five independent actors.

`result.verdict.promote` is only a shadow recommendation: both candidate proposals
must fully pass and at least one baseline must genuinely fail. A baseline ceiling,
partial candidate success or unresolved custody cannot recommend an update. This
protocol does not evaluate performance; optimization projects are rejected until
a separate frozen paired-performance protocol is supplied. Neither successful
controls nor a reviewer approval establishes an improvement by itself.
