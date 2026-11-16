# Paged skill discovery

New proposal buckets are partitioned before native review. Each page has a
content-derived key and its own proposal directory/Pi session. A later page
receives its own structured evidence, not another page's conversation history.
No model merge of all discovery histories is required.

`runReview({discoveryPages:{maxBytes,maxGroups,maxRecords}, ...})` accepts
positive context-validity limits. Defaults are 24 KiB of group payload, four
groups and eight records per page. The limits are frozen on first use and can
be omitted on resume. Elapsed allocation remains the stopping budget.

Original group, project and reference IDs are retained. The persisted
`discoveryPages` manifest records byte counts, continuation counts and every
occurrence/linked-diagnostic reference. Continuation slices are not independent
projects. Linked successful diagnostics supplied by the caller are preserved;
this change does not add success-only projects to the failure inventory.

Oversized records are copied completely to immutable JSON files with hashes.
Their page view points to those files. Long source-string fields additionally
have exact UTF-8 text parts and a line-paged index with field pointers, byte
ranges and hashes. The text directory is declared to the existing native
filesystem tool policy; no extra tool is installed. A directory is an access
scope, while its index and complete JSON are the actual evidence artifacts.
Metadata that cannot fit is rejected explicitly rather than silently dropped.

Completed pages are retained across resume. Existing unpaged completed buckets
and retained native sessions keep their original keys and directories. Changed
record bytes produce different page identities; mutated copied records and
changed frozen limits reject resume. Historical results and elapsed clocks are
not rewritten.

These limits bound discovery group payload. Project facts, baseline guidance,
and dynamic KDB retrieval still require their own compact evidence views; page
limits alone are not a claim that total model context is bounded to 24 KiB.
