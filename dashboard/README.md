# RSI campaign dashboard

This directory is an independent viewer for retained Phase 1 and Phase 2
campaign artifacts. Campaign controls are disabled unless explicitly configured,
and the dashboard never imports campaign execution code. Copy
`config.example.json`, point each root at a locally mounted or synchronized
campaign directory, then run:

```bash
python -m pip install -r dashboard/requirements.txt
python -m dashboard serve --config dashboard/config.json --cache-dir .dashboard-cache
```

The default listener is `127.0.0.1:8080`; use an SSH tunnel for remote demos.
Every refresh is materialized into a SQLite WAL snapshot. If artifact storage
is slow or temporarily unavailable, requests continue serving the last good
snapshot and label it stale. Source is read only through the content-addressed
digests retained by each project; unavailable source is reported rather than
guessed.

## Optional campaign controls

Set `controls.enabled` to `true` only for a loopback listener. The CLI enforces
this boundary and accepts loopback IP literals or the exact `localhost` name;
use an SSH tunnel for remote access. Each preset is a complete server-side A3
launch definition; browser requests can select a preset
but cannot supply paths, commands, environment variables, targets, or arbitrary
arguments. The runner is invoked without a shell. Phase 2 presets additionally
require `phase1_root` and `review`. Every controlled campaign and launch preset
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
while a local Python process still names the campaign root. When historical SFT
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
  --config /path/to/original-runtime.json \
  --legacy-revision 0123456789abcdef0123456789abcdef01234567
```
