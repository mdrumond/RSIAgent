# Host memory supervision

`instance.json` accepts `host_memory: {"max_bytes": 17179869184}`. Missing
configuration defaults to 16 GiB. All instances of the same OS user share
`rsi.slice`; different configured ceilings are rejected. Different OS users
need an administrator-managed common parent limit to establish a machine-wide
ceiling. The module does not claim cross-user enforcement.

Before launching a controller, provision the user slice once while RSI is
stopped. Install `rsi.slice` in the standard systemd user unit configuration:

```ini
[Unit]
Description=RSI aggregate host memory

[Slice]
MemoryAccounting=yes
MemoryMax=17179869184
MemorySwapMax=0
```

Run `systemctl --user daemon-reload` and `systemctl --user start rsi.slice`.
Use the configured byte limit instead of the example when overriding the cap.
Changing a live shared limit is deliberately not automated. An unavailable
user manager, unsupported cgroup v2 memory controller, missing slice, mismatched
cap, or controller outside the slice blocks admission. Standard systemd user
manager delegation must expose the memory controller.

Runtime integration uses `createHostMemorySupervisor(config.host_memory)`.
`controllerCommand(executable, args)` verifies the shared slice and returns
`{command,args}` for launching the entire controller inside a scope there.
Launch KDB as a descendant of this controller, or place its separately managed
service in `rsi.slice`; an existing KDB outside it is not covered. Do not admit
workers until KDB containment is established by the integration.

Inside the controller, `await prepare()` checks its own membership and the
actual kernel limits. `await workerCommand(executable,args,{cwd,env,memoryBytes})` repeats
admission and returns a distinct transient service command with whole-group
cleanup, `OOMPolicy=kill` and `MemoryMax=memoryBytes` (swap disabled).
An explicit positive integer worker cap strictly below the host cap is required.
The scheduler must reserve controller/KDB headroom and ensure the sum of concurrent
worker caps fits the remaining host budget; the launch descriptor does not reserve
capacity across instances. This prevents worker limits from relying solely on
aggregate OOM selection. On service start, integration must verify the actual
worker cgroup `memory.oom.group` is `1` and `memory.max` matches the requested
limit before releasing workload execution; `prepare()` checks the aggregate only.
Pass an absolute cwd and all required environment
entries explicitly: user services inherit the manager environment. Launchers
use `--expand-environment=no` (supported by systemd 255) to preserve literal
`$VAR` and `${VAR}` arguments. Older systemd-run versions lacking that option
will reject launch rather than expand arguments. The
`systemd-run --wait --pipe` process is a launcher, not the worker PID; runtime
identity/reconciliation must follow the worker service. Detaching the launcher
does not stop the service. No worker service is launched by the module itself.

`measure()` returns aggregate `current_bytes`, nullable `peak_bytes`,
`events` (including kernel OOM counters), `max_bytes`, slice name and wall-clock
observation timestamp. Missing required measurements throw instead of producing
RSS estimates. Peak is cumulative over the shared slice lifetime. All subprocess
capability checks are bounded to five seconds. No transport or remote host
containment is added. Runtime wiring and device validation are separate work.
