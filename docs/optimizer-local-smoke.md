# Local add-one optimizer smoke

Use an activated Ascend C runtime on the Ascend host, Node 22 or newer, the Python host dependencies and the pinned Pi installation (`npm ci --prefix integrations/pi`). See [local execution](local-kernel-execution.md). This uses real hardware and the production optimizer, not CPU emulation.

Keep `DEEPSEEK_API_KEY` in your environment or a private environment file outside
the checkout. Create a private worker configuration, replacing the paths,
physical device, and compiler architecture with values for your active runtime:

```json
{
  "execution": "local",
  "workspace": "/absolute/private/pi-smoke/kernel-work",
  "physical_device": 0,
  "arch": "YOUR_COMPILER_ARCHITECTURE",
  "guide": "/absolute/RSIAgent/benchmarks/pi_coding/knowledge/ascendc.md",
  "env_file": "/absolute/private/provider.env",
  "model_profile": "deepseek-flash",
  "max_cases": 1,
  "max_seconds": 300
}
```

This small smoke asks real Pi to generate an add-one kernel for one frozen
64-element FP32 case. It uses the production optimizer, compiler, numerical
oracle, and measurement path, with a five-minute elapsed-time allocation:

```bash
export RSI_SMOKE_CONFIG=/absolute/private/worker-local.json
export RSI_SMOKE_ROOT=/absolute/private/pi-smoke/jobs
python - <<'PYSMOKE'
import json, os
from pathlib import Path
from optimizer.jobs import JobStore, canonical, sha
from optimizer.worker import load_config, run_worker

store = JobStore(Path(os.environ["RSI_SMOKE_ROOT"]))
body = {"files": {}, "memory_entries": []}
memory = {**body, "snapshot_id": sha(canonical(body))}
job = store.create(
    "def forward(x): return x + 1", "forward",
    {"signatures": [{"args": [{"shape": [64], "dtype": "float32"}]}]},
    memory, name="Pi local add-one smoke", max_seconds=300)
result = run_worker(store.root, job["job_id"],
                    load_config(os.environ["RSI_SMOKE_CONFIG"]))
states = [event["payload"] for event in store.events(job["job_id"])
          if event["kind"] == "optimizer-state"]
verified = bool(states and states[-1].get("best"))
print(json.dumps({"job_id": job["job_id"], "status": result["status"],
                  "host_verified": verified}))
raise SystemExit(0 if verified else 2)
PYSMOKE
```

Success requires `host_verified: true`; model prose and compilation alone do
not establish success. Inspect the retained job events, generated source,
operation receipts, and profiler evidence under the private run directories.
Use a fresh root for a new smoke; existing jobs retain their original allocation.
A run may end with `status: "budget-exhausted"` after using its time allocation
and still have `host_verified: true`; inspect these fields separately.
See the [fixed matmul-add smoke](PI_CODING_RUNTIME.md#time-bounded-matmul-add-smoke) for a larger workload.
