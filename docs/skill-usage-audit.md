# Skill exposure and retrieval audit

Run with Node 22:

```sh
node cli/skill-usage-audit.mjs --instance /path/to/campaign
```

The report reads one immutable skill version and a bounded, read-only SQLite
snapshot. `--version` selects a version; `--max-attempts`, `--max-events` (per
attempt), and `--max-retrievals` bound work. Truncation makes coverage incomplete.
A `skill-read` counts exposure, including automatic injection, rather than
proven usefulness. Workflow retrieval evidence remains visible when historical
storage citation recording is missing. Future native Pi retrieval recording
emits an acknowledgement only after citation persistence succeeds; malformed
results are unavailable, rather than empty search results.

Demotion suggestions only propose moving optional, unobserved files to a
searchable tier after `--minimum-opportunities` completed exposures (default 8)
and complete verified telemetry. Canonical workflow scopes and critical guidance
remain protected. No files, skills, KDB entries, active versions, or campaign
configuration change. Unknown unseen KDB inventory is never treated as unused.
The report does not establish whether individual paragraphs within an exposed
file influenced the agent; that needs finer-grained instrumentation or ablation.
