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
