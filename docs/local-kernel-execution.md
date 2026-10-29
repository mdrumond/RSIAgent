# Run Pi kernel work locally

Select local execution in a shared worker JSON file:

```json
{
  "execution": "local",
  "workspace": "/absolute/path/to/kernel-workspace",
  "physical_device": 0,
  "arch": "YOUR_COMPILER_ARCHITECTURE",
  "guide": "/absolute/path/to/programming-guide.md"
}
```

Pass that file with `--worker-config /absolute/path/to/worker.json` to the Pi
smoke, shared Ascend campaign/Phase 1/subject entrypoints, or fixed A3/A5 launchers.
The optimizer worker accepts this spelling and its existing `--config` spelling.
The dashboard uses its existing `optimizer.worker_config` setting. Workflow
configuration still describes model selection, curriculum, guides, and memory;
the shared worker file selects the execution environment.

Activate the desired Python/CANN environment before launching RSI. Pi and its
model gateway run alongside RSI; compilation, correctness checks, timing, and
requested profiler runs execute in local child processes. Local mode does not
invoke cpl-remote, the remote registry, SSH, upload/download, or profile wrappers.
It needs real Ascend hardware and the corresponding installed toolchain; this
is not a CPU emulation mode. Preflight errors stop before model requests.
Device initialization has a 120-second timeout, bounded by `max_seconds`;
`preflight_timeout_seconds` can adjust it for a cold runtime.

For A5 Catlass workflows, also set `catlass_source` to the absolute local source
checkout and `catlass_revision` to its exact commit. The source, native build,
and installed dependencies must satisfy the existing Catlass provenance checks.
Local mode inherits the current environment; it does not install dependencies,
activate Conda, or source an environment setup script. The selected physical
NPU must be allowed by any existing `ASCEND_RT_VISIBLE_DEVICES` mask.

For example, run the fixed Pi smoke using an activated A3 runtime:

```bash
python -m benchmarks.pi_coding.smoke --worker-config /absolute/worker.json \
  --root /absolute/pi-smoke --window local-check --max-seconds 1200 --check-only
```

Remove `--check-only` to run the configured model and kernel workload. Programming
guides and any workflow-specific configuration remain required as before.

Missing `execution` preserves the existing remote path; `"execution": "remote"`
selects it explicitly. Existing remote configurations continue to work. Local
and remote jobs cannot silently switch execution mode on resume. Local jobs
retain host/workspace, device, compiler and toolchain identities, operation
receipts, logs, and evidence. Stop and elapsed-time expiry terminate owned
process groups. Interrupted dispatches without a confirmed process identity
remain uncertain and are not automatically repeated. Device locks serialize
local kernel operations; `lock_directory` can explicitly select a shared lock
location. Keep worker files with machine paths and credentials outside source
control.
