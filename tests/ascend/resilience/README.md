# Orchestration resilience cases

Run `npm run test:resilience -- --tier local --output /fresh/report/directory`.
Supported tiers are `local`, `deepseek`, `remote`, and `all`; select `--case ID`
for replay, `--seed INTEGER`, and `--profile normal|slow|bursty`. The default
round allocation is 5400 elapsed seconds including setup and transport. The last
300 seconds are reserved for cleanup/reporting; actor allocations remain separate.
All required tiers must be registered for an unfiltered `all` run to pass.

Case modules named `*.case.mjs` export `cases`, an array of unique `{id, tier,
async run(context)}` entries. Context contains `directory`, `signal`, `seed`,
`profile`, `event(value)`, and `onCleanup(callback)`. Every case must propagate
the signal into native operations, register task-owned cleanup before execution,
and assert observable expected outcomes before returning evidence. A safe held
attempt is a successful safety scenario only when the case asserts that outcome;
it is never represented as a successful project. Cleanup errors fail the case.
Missing credentials/registered transport use `Blocked`, never a passing skip.

Keep schedules, native sessions, logs, receipts, and closure proofs under the case
directory. Never log credentials or change a production instance. DeepSeek runs
on the controller; target workloads are Python CPU programs. Remote cases use
only the installed CPL interface and user registry, with no NPU work or shared
network/process/service changes. Before every native remote operation read the
installed remote-access instructions.

`schedule()` supplies seeded 10 ms normal latency, 250 ms slow latency, or seeded
one-second bursts. Test-only faults select a named boundary and occurrence, for
example `{point:'run.after-receipt', occurrence:1, disconnect:true}`. `target()`
executes real SDK local Bash operations in a distinct target tree using the
production environment's existing client injection. `serve()`/`connect()` allow
separate observer processes to share that retained backend. They are fixtures,
not a replacement production transport or process/session manager.
