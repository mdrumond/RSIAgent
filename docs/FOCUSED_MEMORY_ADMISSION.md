# Focused memory admission

Subject campaigns may set `memory_admission_mode` to `focused` and
`actor_memory_delivery` to `kdb-only`. The default remains `paired`, preserving
existing full Actor A/B admission and its persisted evaluations. Select a new
campaign root when changing modes.

Source review still checks concrete claims, exact source snippets, applicability
and retained evidence. Focused admission then freezes a related transfer contract,
replays the full retained before/after sources against the source project's host
oracle, and executes one fresh Actor with candidate memory available through KDB.
The related contract must differ in workload, not just its project ID or prose.
The independent judge inspects the sources and result/tool logs and must establish
explicit retrieval and application of the lesson. A passing before-source does
not demonstrate a repair. Missing before-source remains a source-reviewed
reference; it is not published as a validated repair lesson.

Each stage persists its start before calling external work. Completed judgments
are immutable on resume; an interrupted stage is held rather than resubmitted.
The normal admission gate still owns rewriting, stale-base checks and publication.
It retains `mode`, `claim_kind`, and `evidence_stages` with admission provenance.
Infrastructure failures and elapsed deadlines remain inconclusive.

Correctness and transfer do not establish performance. An optional production
composition binding `measure_memory_pair(request, before_sha, after_sha,
directory, remaining_seconds=...)` may return host-owned repeated timing evidence.
The report must contain the exact `runtime`, SHA256 of canonical source `request`
as `contract_sha256`, `sources: {control: before_sha, treatment: after_sha}`,
`host_verified: true`, and at least three `pairs` of positive finite `control_us`
and `treatment_us` device durations under identical inputs/runtime conditions.
The measurement backend owns healthy idle isolation, timing semantics and retained
evidence. The independent judge assesses whether those measurements support the
actual claim. Without that binding, performance claims require rewriting or remain
inconclusive; wall-clock compilation/Actor times are not device speedups.
