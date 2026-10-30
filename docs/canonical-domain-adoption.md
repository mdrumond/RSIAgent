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
