# RSI campaign dashboard

This directory is an independent, read-only viewer for retained Phase 1 and
Phase 2 campaign artifacts. It never imports campaign execution code. Copy
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
