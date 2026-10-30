# Canonical domain adoption contract

`rsi.domain_execution.DomainRunner` composes the existing canonical
`WorkflowActor`, `Workflow` and `StrategyEvidence`. It inherits the production
runner's control/reconciliation loop and evidence tools. It introduces no
provider loop, KDB or remote transport.

A target binding supplies an existing host executor through these methods:

- `identity`: frozen target, provider, KDB, memory and verifier identities. Include
  exact target/runtime/compiler revisions, case-suite identity and measurement
  contract in this data. Credentials stay outside it.
- `store`: the existing content-addressed evidence store; `editable_files`:
  allowed source paths. `read(path)` returns content and whole-project hash.
- `project_hash()` and exact-hash `edit(...)`: preserve the target source contract.
- `preflight(context)`: perform host setup and check frozen target capabilities.
- `execute(stage,args,operation,activity)`: perform one canonical stage and return
  a trusted `DomainObservation`. It must not call a legacy actor/search loop.
- `context()`: the existing admitted domain observations and memory presentation.

Pass the existing registered Pi provider through `session_factory` and the
existing configured KDB through `query`. The KDB callback returns retained
positive results, negative records and provenance, using the existing
`knowledge_db_query` adapter or an equivalent adapter over the domain's existing
knowledge service. Empty results are valid; an unavailable KDB is a suspension.
Skill bundles and step maps are frozen before model work. Design/optimisation
must actually retrieve, retrospect and select against the returned evidence.

`DomainActivity.run` receives a synchronous host callback which observes its
operation to completion. Use a distinct stable name per actual compiler,
execution or profiler operation with `execution=True`. The callback receives
remaining actor/wall allowance and must propagate it to the existing bounded
host runtime. Failed results consume their reservation and elapsed allocation.
An exception does not prove remote completion: the interval remains unresolved
until the controller obtains an authoritative closure receipt. Resume never
redispatches an unknown operation. Setup and actual remote data copying use
separate `setup`/`transfer` activities. Do not wrap those activities inside a
billable interval. Remote-clock epochs must never be used to subtract time from
controller timestamps; use the existing verified remote elapsed receipt path.

Test and validation require actual admitted host executions. Validation must
return an explicit frozen-oracle correctness outcome, independently of its
performance outcome. Acceptance requires current-source successful validation.
Source changes outside a pending operation require explicit reconciliation.

`CanonicalAdoption(build,finish,distill)` binds this composition into the public
A3 trial, A5 direct/Pi/CoreAttemptDriver and optimizer entrypoints. `build`
returns a `DomainRunner`; `finish` translates its result to the existing typed
public result and retains the target ledger, measurements and memory history.
The optional terminal distiller must consume this exact retained history and
charge genuine provider/retrieval work to the same allocation. These callbacks
must not invoke `_run`, `_run_owned` or another old programming loop.

The public entrypoints require this binding explicitly. They raise
`CanonicalConfigurationError` when it is missing; they do not silently fall
back to the old programming workflow. Existing production constructors must be
wired to the target-specific factory before enabling those paths. This adapter
is an adoption seam, not evidence that production device bindings have already
been migrated or validated.

CPU stub-host tests exercise the real canonical workflow, numerical repair,
actual skill/KDB receipt admission, elapsed accounting and pending-operation
behavior. They establish no device/compiler/profiler capability.

## Default production bindings

DirectActorDriver/PiActorDriver now select the shared kernel-action binding by
default. It calls the existing ordinary/adaptive host compile/run/measurement/
submit handlers per canonical stage, preserving their admission and numerical
verification gates. It uses the configured Knowledge Agent through its existing
action protocol and retains both its full result index and actual failures.
Terminal learning uses the existing registered provider and distillation parser,
with provider work charged to the same canonical allocation.

A3TrialLoop defaults to its native candidate/compiler, profiler, Knowledge Agent
and memory journal binding. The existing registered Pi session supplies canonical
actions. Prepared oracle and host verification remain target-owned. Explicit
recovery-policy wrappers expose the same underlying actor and starting source.

OptimizationLoop defaults to frozen-reference preparation, its existing bounded
backend evaluation/profile methods, score verification, and existing job/history
store. Prepare-only pauses at the task-construction boundary. A subsequent
programming project links the existing frozen suite and carries the remaining
parent allocation. Single-evaluation projects perform one backend candidate
evaluation and validate the retained result against the frozen suite locally;
they keep the enclosing job available for its next explicitly requested project.
Cumulative canonical elapsed receipts are idempotent across resume.

These defaults replace the previous requirement that every production caller
inject a factory. Custom CanonicalAdoption remains supported. An enabled baseline
KDB is required by the new mandatory retrieval contract; a legacy KDB-disabled
treatment fails clearly rather than being relabeled as a successful empty query.
CoreAttemptDriver has no active production constructor and still requires an
explicit provider/baseline binding.

Existing transport upload/copy operations now expose controller-local excluded
phase scopes. They pause only their own enclosing operation interval; another
concurrent billable interval remains charged by the canonical union clock.
Historical transport evidence and independent bounded transport wall guards are
preserved. No remote-device validation is implied by these source adapters.

## Observed native failure/export and rehydration

Domain contract v2 persists its existing CAS location relative to the task.
DomainActivity retains a scoped dispatch intent before host work and a typed
result/accounting receipt only after its actual callback observation. A returned
pending/running/unknown status is not completion: it suspends the same operation
and retains timing uncertainty. Exceptions remain unobserved unless the host
explicitly guarantees a synchronous local source-edit failure. No new store or
actor loop is introduced.

DomainRunner's host-neutral completed-effects adapter validates real native
reservations/results, target/provider/KDB/skill/budget identity, retrieval
completion and elapsed intervals. The existing parent terminal_failure_proof and
trusted terminal observation APIs consume that adapter, preserving pending stage
and source evidence. Unknown remote work and unresolved timing cannot gain a
terminal failure proof or export admission.

The shared evidence_io.export_task recognizes domain tasks and uses a read-only
normalized input adapter. The same existing serializer and evidence/SFT schema
retain raw CAS bytes and event order. The input view declares a domain contract,
not fabricated Python commands or an invented verifier. Original immutable CAS
objects are linked into temporary export staging (copied only across filesystems)
and never modified. No second logical evidence database is created.

Host snapshots retain typed, host-owned compilation/verification/final-run/profile
outcomes, source bytes, cases and provenance. The codec supports registered native
dataclasses/enums and validates derived fields on reconstruction; it never uses
pickle, evaluates repr strings, or runs candidate source. Accepted checkpoints
must remain current-source verified. Fresh wrappers restore those actual CAS
outcomes and return retained public results without another provider/compiler/run
or duplicate memory publication. Existing native journal records are checked
before committing memory. Old checkpoints lacking reconstructible typed native
outcomes require explicit migration; success is never inferred from text alone.

Failed preflight and task construction are terminal rejections in the shared
Workflow. They retain their failed-stage evidence and host admission failure;
they do not fabricate an Acceptance stage or retry unbounded excluded setup.
