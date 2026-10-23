# Analysis-only KDB pilot

`benchmarks.ascend.kdb_evaluation` compares curriculum, code-writer, and
performance-analyzer decisions using retained evidence. It exposes no execution
tools and makes no claim about executed kernel correctness or speedup.

Freeze 18 `Case` entries (six per role; analyzer split three interpretation and
three optimization), their evidence hashes, and evidence-backed rubrics before
calling providers. `validate_cases` checks counts, evidence integrity, and
family/content separation from the producer snapshot. Curate related attempt
families explicitly: names or distinct source hashes alone do not prove independence.
Keep rubric answers and target repair outcomes outside all retrieval collections
and agent prompts. Never fabricate missing profiling evidence.

The four arms are A: no retrieval or optional memory; B: reference retrieval;
C: reference retrieval plus the full compact snapshot supplied directly; D:
reference and memory retrieval. `memory_documents` must be the exact string from
`MemorySnapshot.direct_documents()` used by the memory collection in D. Every
arm gets a new provider conversation and fresh selected-collection gate.

## Integration recipe

Prepare an ignored driver using the registered runtime and the ordinary project
provider configuration. Inject `complete(messages, seconds) -> {text, tokens}`.
It must pass the full messages to the frozen DeepSeek profile, enforce the
provided HTTP timeout (including connection and response), disable SDK retries
that exceed the allowance, and map SDK timeout exceptions to builtin
`TimeoutError`. Do not use the default unbounded provider transport.

Inject `gate_factory(arm)`. Its object implements
`query(KnowledgeQuery, seconds) -> tuple[KnowledgeResult, ...]`,
`read_artifact(collection, digest, offset=0, limit=16384, seconds=...)`,
and optional `close()`. B/C select reference manifests only; D also selects the
same frozen memory snapshot. Artifact reads must use the snapshot's hash gate.
Bound retrieval by the supplied elapsed allowance too; large embedding scans
can otherwise exhaust an arm before the next provider call.

```python
rows = json.loads(case_manifest.read_text())
cases = [Case(**row) for row in rows]
validate_cases(cases, producer_families, producer_artifact_hashes)
identity = digest({"cases": rows, "model": model_profile.fingerprint,
                   "manifests": frozen_manifest_fingerprints,
                   "memory": snapshot.fingerprint, "runner": runner_revision})
budget = Budget(output_dir / "state.json", identity)  # 1800 elapsed seconds
pilot = Pilot(complete, gate_factory, snapshot.direct_documents(), budget)
report = pilot.run(cases)
(output_dir / "report.json").write_text(json.dumps(report, indent=2))
```

Use the actual snapshot identity field supplied by the snapshot manifest in the
example's `memory` entry. Credentials and machine routes stay in ignored user
configuration. The driver and case artifacts may be retained in ignored task
runtime storage; never publish credentials or raw provider exceptions.

Run serially. Each arm receives at most 45 seconds; each blinded case-group
judgment receives at most 30 seconds, always within the global allocation.
Resume using the identical frozen identity and state file. Interrupted active
work is charged across restart; a clean pause preserves remaining allocation.
Do not reset the elapsed budget to complete more cases.

The state retains arm transcripts, attempted queries, validated retrieved
citations, declared numerical-claim checks, timing and token telemetry, blinded
judgments, and incomplete results. The report excludes incomplete four-arm or
judgment groups from paired comparisons and reports each role separately.
Structured numerical checks cover only declared frozen facts; the blinded
rubric must also assess numerical claims in prose against supplied raw columns
and units. Six cases per role provide exploratory evidence only.

Numerical-check fields distinguish registered fact mismatches
(`unsupported_numerical_claims`) from identifiers outside the automated fact
registry (`unregistered_numerical_claims`). An unregistered claim has no automated
correctness verdict; the judge checks it against raw report evidence and must
not penalize it solely for lacking a registered identifier.

For a separately identified extended cohort, pass `arm_seconds=180` and
`judge_seconds=60` to `Pilot`; defaults remain 45 and 30. Both accept positive
finite seconds. Each nested allowance remains capped by the remaining global
`Budget`. Record cohort settings in the frozen identity and preserve prior
cohort outputs and elapsed allocation when extending an explicitly authorized
campaign.

## Six-condition adaptive retrieval evaluation

Pass `adaptive=True` to `Pilot` to select six conditions; this flag selects the
comparison design, not the implementation of injected gates. A remains no KDB;
B uses corrected deterministic reference retrieval; C adds the exact complete
direct-memory bundle to B; D adds indexed memory to deterministic retrieval;
E uses the adaptive librarian over references; F adds indexed memory to E.
The factory must enforce those selected collections. Only C receives the direct
bundle, and only D/F may access the frozen memory snapshot. Each case rotates
all six arms, producing 108 fresh conversations for the frozen 18 cases.

A gate can optionally implement `configure_context(task, role, context)`.
The harness calls it once with the ordinary case prompt, role, and public
case context. No rubric is sent to the gate. Target/runtime information belongs
in that ordinary task/context. The gate owns a shared 90-second retrieval
allowance across its queries and artifact reads, bounded by each supplied
remaining-arm allowance. Provider calls inside the librarian must use the
same actual DeepSeek profile as live validation; mocked unit tests alone do
not validate the adaptive agent. Retain librarian decisions and provider
telemetry in the ignored run evidence.

Use a separately frozen identity with `adaptive=True`, `arm_seconds=180`,
`judge_seconds=60`, and a fresh 7200-second elapsed `Budget`. Keep all original
cohort state unchanged. The new design grades each completed answer independently
against the same rubric, shared context, exact full memory, retrieved evidence,
and objective checks. Every judge packet uses the unrelated candidate ID `0`;
it contains no arm label. Grade order is deterministic and unrelated to treatment
order. Sixty seconds is shared across all six judgments for one case, not renewed
per answer. Successful grades survive other answer, judge, or transport failures.

State records `judgments[case_id].grades[arm]` with its status, raw request,
response when returned, score/reason, tokens, or error type. Interrupted judging
is charged to the same persisted case allowance on resume. A previously attempted
grade is never silently retried; missing grades may continue only within the
remaining original allowance. Run identity must include model/provider identity,
selected manifests, snapshot, source revision, arm settings, and this comparison
mode to prevent accidental mixing with four-condition state.

Six-condition reports retain partial per-arm scores. Each comparison carries
its own `cases` and `matched` denominator, using only cases with valid answers
and grades for both treatments. Primary comparisons are E−B (adaptive references),
F−D (adaptive indexed memory), D−C (indexed versus direct memory), and F−E
(memory contribution with adaptive retrieval). B−A, C−B, and D−B remain available.
Role-level `matched` counts fully graded six-arm groups as a secondary statistic;
it does not determine eligibility for individual paired comparisons. Report
invalid outputs, incomplete grading, abstention, retrieval relevance, citation
errors, latency, and tokens separately from score differences. These small,
model-judged samples do not establish statistical effects or kernel speedups.
