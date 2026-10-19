# Optimizer loop backend contract

`OptimizationLoop(store, job_id, agent, backend, knowledge=callable).run()` is
backend-neutral. The detached worker owns process exclusion, runtime selection,
hard operation timeouts, transport handles, and interruption observation.

* `prepare(spec, directory)` runs and freezes Torch first. Return a nonempty
  `suite_sha256` and `cases` with unique `case` identities, native `contract`, and
  `timing.torch.latency_us`. Preserve frozen manifest/artifact provenance. Capture
  baseline rich diagnostics here; unavailable diagnostics must remain explicit.
* `evaluate(source, directory, block_count=...)` returns `native.evaluate` JSON
  using the frozen baseline, `benchmark=True`, `profile=False`. Compilation or
  execution failure is a retained non-verified result, rather than an exception.
* `profile(source, directory, block_count=...)` returns rich compact diagnostics
  for a newly best candidate. Never transfer timelines.

All three calls are idempotent for their directory: atomically retain the native
handle before observing it; on recovery observe that handle, never resubmit an
uncertain operation. Retain completed result JSON and return it on repeated calls.
The candidate directory is the SHA256 of UTF-8 source; profile is its subdirectory.
Enforce remaining job wall budget in the worker's operation timeout. The loop
checks budgets and Stop at action boundaries and charges elapsed provider, prep,
evaluation, profiling, and KDB time. In-flight downtime counts on recovery.

Job events contain immutable state checkpoints and digest-authenticated artifacts.
Submission signatures use `{"signatures": [...]}` as the dashboard contract;
legacy CLI lists and single-case objects remain accepted. Source artifacts are
canonical JSON string bytes; decode JSON to recover the exact UTF-8 source for
display/download. The artifact digest authenticates the stored JSON bytes.
Resume uses `JobStore.resume` with explicit additional attempts/time. It preserves
the reference submission, selected memory, suite, model identity, and source
history. The final reserved attempt is evaluated even though its reservation
exhausts the attempt count. Profiling after an exhausted budget is marked
unavailable. Slower correct candidates remain usable results.

`OptimizerAgent` defaults to the canonical A3 GPT-5.6-Sol profile. The composition
must supply the programming guide and callable KDB query tool. Actual requests,
responses, token usage, and tool results are retained. The protocol accepts brief
decision summaries, never requests private chain-of-thought.
