# RSI campaign dashboard

This directory shows current subject campaigns alongside retained Phase 1 and
Phase 2 artifacts. Campaign controls are disabled unless explicitly configured;
new Phase 2 execution paths are retired. Copy
`config.example.json`, point each root at a locally mounted or synchronized
campaign directory, then run:

```bash
python -m pip install -r dashboard/requirements.txt
python -m dashboard serve --config dashboard/config.json --cache-dir .dashboard-cache
```

## Kernel Optimizer

The Kernel Optimizer tab is always visible. Configure its independent
`optimizer` block using `config.example.json`: an absolute job `root`, explicitly
listed native A3 Phase 2 `memory_roots`, reviewed seed `snapshot_roots`, and an
existing absolute ignored `worker_config`. Set `enabled` only when the optimizer
worker is installed and configured. Enabled optimizer controls require a
loopback bind, independently of campaign controls.

Paste Python or upload a `.py` file, select the function/class entrypoint, enter
representative input signatures and optional constructor arguments, and choose
eligible completed memory. Discovery shows unavailable sources with their reason;
creation authenticates the source again and pins its snapshot. Defaults are 25
candidate attempts and 7200 seconds. Resume requires explicit additional budget.

Jobs run in detached `python -m optimizer.worker` processes. The dashboard does
not execute submitted Python and restarting Flask does not launch jobs again.
Worker output remains in each job's `worker.log`; durable manifests, authenticated
events and immutable submissions supply status and lazy trace details. Lifecycle
POST requests require a CSRF token and persistent `Idempotency-Key`; repeating the
same action after a dashboard restart does not add budget or launch twice.

The default listener is `127.0.0.1:8080`; use an SSH tunnel for remote demos.
Every refresh is materialized into a SQLite WAL snapshot. If artifact storage
is slow or temporarily unavailable, requests continue serving the last good
snapshot and label it stale. Source is read only through the content-addressed
digests retained by each project; unavailable source is reported rather than
guessed.

Project hypotheses that use the curriculum's ordered `Question`, `Controlled
change`, `Tools`, `Measurements`, and `Expected lesson` convention are rendered
as labeled fields; historical free-form text remains unchanged. Project detail
sections are collapsible, and the final section indexes the complete retained
curriculum, actor, host-verification, and outcome trace. Trace payloads are not
included in page HTML or `/api/status`: opening an event fetches its exact bytes
through a project-scoped identifier. SHA-256 is checked before the immutable
payload is added to the SQLite cache. UTF-8 payloads are returned as escaped
text, while every retained payload is available through the raw download route.
Missing prompts or artifacts are explicitly marked unavailable.

Fresh autonomous Phase 2 campaigns appear before learner `state.json` exists
when `goal-selection.json` publishes selection or reference-qualification progress.
Only its public candidate specification, rationale, citations, proposal count,
diagnostic and qualified target ID are shown. Reference qualification is not
counted as learner verification; private reference source and conversations are
never indexed by the viewer.

Target-driven Phase 2 rows and API routes use `attempt-NNN--project-id`, so target
retries retain distinct source, traces and performance. Existing Phase 1 and
legacy Phase 2 project routes are unchanged. The project-wide host ledger cannot
distinguish target retries; these rows expose each branch's exact verifier result
instead of attributing another attempt's ledger entries. Terminal readiness,
stalls and budget exhaustion take precedence over an old worker status; running
target/practice/review stages and worker failures remain visible separately.

Opening a project's performance section lazily loads retained timing and
`PipeUtilization` facts. The local SVG views show attempt distributions, shape
sweeps, and AIC/AIV pipe ratios, with the exact normalized values in an HTML
table. Useful logical-I/O bandwidth is a derived value based on declared tensor
bytes and ordinary launch-plus-synchronize latency; it is not device MTE
bandwidth. Missing metrics are shown as `NA`, distinct from a measured zero,
and MTE bandwidth remains unavailable unless a future campaign retains a
dedicated `Memory` profiler capture.

## Optional campaign controls

### Failure-directed learning

Set top-level `feedback_root` to the shared Phase 2 feedback directory. The
**Failure feedback** tab displays immutable original outcomes separately from
investigation status, classifications, linked projects and retries. A verified
but slow kernel remains a correctness success with a performance shortfall.
Evidence previews authenticate the retained digest on the host and return only
an 8 KiB prefix; they never fetch remote raw profiler trees.

With controls enabled, **Import configured Phase 2 results** runs the historical
importer against every Phase 2 campaign in this dashboard configuration. Select
failures, a compatible fresh A3 destination preset, and an explicit project
budget to start a retained Phase 1 investigation awaiting curriculum approval.
New Phase 2 launches and original-target retries are rejected before dispatch;
use subject curriculum requests for new work. Phase 2 sampling, approval, and
execution actions are unavailable, while historical proposals and results remain
readable. Explicit Phase 1 investigations use the asynchronous operation queue;
no campaigns start merely by viewing or importing feedback.
Resolved results and deferred/infrastructure observations remain visible but
are not selectable for kernel investigation. Each Phase 1 investigation needs
unused preset root/control directories, just like retained campaign starts.

Set `controls.enabled` to `true` only for a loopback listener. The CLI enforces
this boundary and accepts loopback IP literals or the exact `localhost` name;
use an SSH tunnel for remote access. Each preset is a complete server-side A3
launch definition; browser requests can select a preset
but cannot supply paths, commands, environment variables, targets, or arbitrary
arguments. The runner is invoked without a shell. Retained Phase 2 configuration
is readable but cannot authorize new launches. Every controlled campaign and launch preset
must explicitly set `"target": "a3"`; other targets remain viewable but cannot
be controlled.

Mutation requests require the page's CSRF token, an explicit confirmation, and
an idempotency key. They are serialized by one background worker and persisted
in the same SQLite WAL database as the read cache. A dashboard restart marks a
queued or in-flight operation interrupted rather than submitting it again. The retained
campaign approval record remains authoritative for a deliberate retry.
Repeating the same idempotency key returns the original operation; use a new key
to explicitly resume or reconcile a retained `running` approval after an
interrupted command. Operation details are unavailable when controls are
disabled, including when the read-only dashboard is intentionally bound beyond
loopback.

The proposal view exposes the hypothesis plus deterministic novelty and coverage
comparisons. A terminal proposal exposes only its status and concise reason and
can be approved without selecting a project. Provider messages and hidden
reasoning are never copied into the cache or API response.

Completed legacy A3 Phase 1 campaigns can be adopted without modifying their
retained history. The adoption command requires the original seed, runtime
configuration, a source manifest for future work, and the exact legacy harness
revision; it atomically creates a separate control directory and refuses to run
while a local Python process still names the campaign root. Absolute and relative
`--root` forms and retained `_campaign --control` workers are recognized through
their process working directories and authenticated job metadata. The recovery
supervisor must also be stopped or disabled before adoption; the command checks
again immediately before publishing controls but never terminates processes.
When historical SFT
trace headers exist, their model and source-manifest provenance is validated.
If their digest differs from the supplied manifest, both digests are recorded
separately and the supplied manifest remains scoped to post-adoption work. When
no historical headers exist, the adoption record explicitly marks historical
source provenance unavailable. The supplied manifest is never presented as
evidence for old projects. After configuring that
control directory in the dashboard, **Extend by 8** is
available only at a clean `TARGET_REACHED` boundary. It invokes continuation,
but never samples or runs a project.

```bash
python run_ascend_campaign.py adopt \
  --control /path/to/external-control \
  --root /path/to/completed-phase1 \
  --objective "Learn A3 kernels" \
  --initial-memory /path/to/original-seed.bin \
  --config /path/to/original-native-a3.json \
  --model openai/gpt-5.6-sol \
  --source-manifest /path/to/post-adoption-sources.json \
  --knowledge with-kdb \
  --profiling without-profiling-guidance \
  --legacy-revision 0123456789abcdef0123456789abcdef01234567
```

Subject campaigns can be configured alongside historical campaigns with
`subject_campaigns: [{"id": "research", "name": "Research", "target": "a3", "root": "/path/to/subject-campaign"}]`.
Their project pages separate execution, project review, memory admission, and
publication. Actor logs follow retained trace, provider-evidence, and optimizer
ledgers every two seconds. Pause stops polling; loading earlier entries reads
from the beginning without duplicating displayed records. Downloads include
complete retained records at download time. Partial trailing records appear
only after their writer finishes the line. Interrupted writers are shown as
stopped when the campaign execution lock is no longer held.

Enable curriculum assistance with `controls.enabled: true` and add server-owned
`config` (subject runtime settings JSON) and `initial_memory` paths to each
subject campaign entry. The retained runtime/settings fingerprints must match.
Formatting runs asynchronously under the explicit elapsed allowance entered in
the form. The curriculum returns an executable contract and a diff; only approval
changes future subjects or the unreserved project queue. Reserved wave contracts
stay frozen. Project restarts additionally require a fresh execution allowance
and dispatch the approved prompt through the durable actor-control service.
Restart preview and approval require the subject interaction runtime (provided by the actor-controls change); installations without it retain read-only restart history and reject dispatch before provider work or approval changes.
Role bindings are isolated from live campaign/admission workers. Existing
operation polling, CSRF checks, and idempotency apply to previews and approvals.

Approved restart proposals retain a Resume/reconcile action after a launch
failure or dashboard interruption. The request uses the same durable identity
and original elapsed allocation; queued/running requests are observed rather
than launched again. Status continues polling after the dashboard operation
finishes. Completed requests and uncertain executions requiring observation
cannot be retried from that action. The editable future-work queue excludes
projects already reserved into a wave, even if they have not started.

Live-log cursors report resets per stream, preserving unaffected entries when
one writer rotates or truncates its ledger. Complete records larger than 1 MiB
are replaced by an explicit preview-limit entry; the cursor advances so later
messages continue to arrive, and the full record remains in the log download.
Unfinished trailing records retain their start even when larger than the preview
limit and are consumed only after the writer appends their newline.
