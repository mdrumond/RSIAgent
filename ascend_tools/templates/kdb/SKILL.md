---
name: ascend-knowledge
description: Retrieve indexed source evidence and record compact Ascend lessons.
---
Search both good and negative knowledge for the current target during design
and optimisation. Cite collection, chunk ID, path, line range, and source hash.
Reflect on relevant prior attempts before proceeding. Record concise lessons
with their target applicability, outcome, attempt ID, and direct evidence file
references or native indexed entry IDs. Never copy raw_refs or archived traces
into lessons. A returned result is retrieval evidence, not device validation.
Use returned evidence references directly when recording lessons. Indexed
references remain valid after original source checkouts disappear. A lesson's
delivery_id identifies idempotent delivery; its references list contains the
actual indexed collection/entry IDs to cite in later lessons.
Use read_entry(collection, entry_id) to recover actual stored text from prior
retrieval citations when the original checkout is absent. Check truncated and
total_bytes; returned evidence remains a direct citation to the full chunk.
