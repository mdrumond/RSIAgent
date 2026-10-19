# Detached A3 optimizer worker

Run from the task checkout with its Python environment:

```bash
python -m optimizer.worker --root /absolute/jobs --job-id JOB_ID --config /absolute/ignored-worker.json
```

The dashboard launches this argv with `start_new_session=True`, appending stdout
and stderr to the job's `worker.log`. The worker outlives the web process; page
loads and web restarts must not launch replacements. A per-job nonblocking flock
rejects duplicate workers. A shared device flock queues different jobs until the
configured device is free. Stop also interrupts queued device admission.

The ignored JSON config requires `profile` and matching `target` (checked A3
profile), `runtime` (registered Torch runtime), numeric `physical_device`, `arch`,
and absolute `cpl_remote`, `validation_wrapper`, `remote_workspace`, and `guide`
paths. Discover the target/runtime/workspace and device through current registry
preflight; confirm the configured remote workspace already exists before launch.
The architecture is the discovered compiler architecture, not a device nickname.
Optional `env_file` loads provider credentials only in the detached worker.
`model_profile` defaults to the existing A3 OpenAI profile. Optional `knowledge`
contains `knowledge_manifest`, `knowledge_database`, and `embedding_cache` paths;
the existing read-only manifest-checked KnowledgeAgent gates every query.
For the current expanded multi-collection native campaign, use `knowledge` with
`kdb`, `bge_cache`, `collection` (string or list) and `manifest` (the pinned
fingerprint string or selected-collection fingerprint map). This uses the
existing native campaign KnowledgeAgent and authenticates every selected manifest
before the provider loop starts.
Optional `lock_directory` coordinates optimizer workers sharing a device. Its
default is `.optimizer-locks` beside the configured validation wrapper. Existing
campaign/device admission remains an operator prerequisite.
Optional `max_cases` bounds the frozen suite to 1–24 cases (default 24); supplied
representatives are never discarded, so a bound below their count is rejected.

Each backend method is idempotent for its local execution directory. Source and
request packages upload through cpl-remote; initial native dispatch uses the
checked neutral wrapper with explicit runtime/device. Observation uses the
checked A3 profile adapter's five-second `--resume` path. Native/domain handles
are retained in job events and `transport.json`. Stop or the job deadline cancels
the exact retained native handle. A dispatch sentinel with no returned handle is
`observation-unavailable`: inspect/recover the original operation, never delete
the sentinel or resubmit to guess its outcome. Transfer handles are independently
retained and observed/fetched without duplicate submission.

Reference preparation freezes reproducible Torch inputs/outputs remotely before
canonical CANNBot baseline timing. Tensor payloads remain there. Separate richer
diagnostics are attempted for Torch and best verified native candidates. Only
compact summary CSV/JSON, compilation/source metadata, and logs are downloaded;
timeline reports and shared libraries are excluded. Downloaded artifacts are
digest-checked and stored in the job artifact store. Every remote runtime call is
bounded by the job's remaining wall-time budget. Runtime/device/workspace/guide
identity is pinned across resume.

For a bounded production proof, create a job using the normal JobStore interface,
then call `run_worker(root, job_id, config, agent=stub)` with an injected agent
whose identity is fixed and whose `complete(stage,context,record)` returns cases,
one Ascend C candidate, and finish. This still executes the actual production
transport, freezing, compiler, CANNBot timing, profiling and hydration. Repeat
without `agent=stub` for the configured real provider.
