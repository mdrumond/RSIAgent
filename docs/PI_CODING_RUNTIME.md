# Native Pi runtime

Pi 1.3.0 owns model requests, conversation persistence, native filesystem tools,
retries and compaction. Python owns the canonical workflow, elapsed budgets,
knowledge permissions, device leases, execution, frozen correctness cases and
Memory admission. Desktop and OSWorld agents are unchanged.

## Installation

Install the packaged Python CLI and the pinned Node dependencies with Node 22
or newer:

```sh
python -m pip install .
npm ci --prefix integrations/pi
```

For independent instances, use [instance setup](rsi/quickstart.md). Setup verifies
the native SDK exports and retains runtime identities before allowing execution.
Keep provider credentials in the launching environment or ignored configuration.
Missing runtime dependencies fail before provider dispatch. There is no direct
HTTP completion fallback and no separate NOOA Python worker requirement.

## Canonical execution

`ProjectRunner` remains the canonical stage coordinator. The native session
executes domain tools and returns their actual host observations. Native read,
edit and write tools operate on admitted paths. Host source guards retain
pre-write intentions and resulting hashes; a stage commits only after verified
native settlement and the host's evidence checks. Unknown effects require
observation rather than blind replay.

Native skills come from the frozen bundle. File-backed reference resources keep
their relative paths. Large observations are retained in immutable CAS objects
and exposed through readable paths and useful indices. Native Read handles line
pagination. Python does not rewrite conversation history or implement a second
compaction engine. Configured operator guidance reaches the canonical context;
stale queued source actions are rejected when that guidance changes.

Programming curriculum, terminal distillation and review use typed native tools.
The host still validates proposals, citations, numerical evidence and negative
knowledge. A model proposal cannot establish successful execution or correctness.
Explicit scripted sessions and injected transports are test seams only.

## Timing, closure and retention

Stopping allocations use elapsed time. Setup is recorded separately; provider
waits, native retries, compaction and host activity retain their measured charges.
Request, token and candidate counts remain telemetry. Resume preserves elapsed
spend and unresolved operation handles.

Each project retains its native session and raw event journal. High-frequency
partial observations carry byte ranges and digests into that journal, avoiding
another copy of every cumulative partial in host CAS. Closure retains the whole
journal in CAS and maps its filename to the immutable digest. Lifecycle, usage,
fault and settlement observations remain immediately available to supervision.
SFT export retains the evidence closure. Unknown provider or tool settlement is
never converted into successful closure.

## Behavioral validation

```sh
python -m pytest -q tests/test_native_pi_session.py tests/test_native_pi_filesystem.py
python -m pytest -q tests/test_native_pi_workflow.py tests/test_native_pi_programming_roles.py
python -m pytest -q tests/test_native_pi_composition.py tests/test_rsi_domain_distillation.py
```

These tests use real SDK sessions against local HTTP fixtures. They exercise
canonical stages, native source edits, numeric failure and repair, compaction,
operator steering, typed role feedback, uncertain settlement and retained
exports. They do not establish NPU correctness. Historical device evidence is
separate in [PI_CODING_EVALUATION.md](PI_CODING_EVALUATION.md).
