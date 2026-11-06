# Closed evaluation KDB archives

Fresh finite campaigns set `skills.archiveEvaluations=true`. Other instances keep
their existing behavior. A completed comparison is recorded once in the parent
SQLite `evaluation_archives` queue with its originating completed Phase 2 attempt.
Status includes queued/completed archives and the last error. Status reads do not
create queue entries. A completed comparison missed by an interrupted enqueue is
discovered again before the next background comparison.

The background pipeline drains archives before generating another candidate. It
requires settled/postprocessed parent and sample attempts, stopped child controllers,
workers, operations, storage requests and KDB service owners. Unknown or incomplete
comparisons remain fenced. Frozen input instances must also have stopped writers.
Only ordinary, unaliased KDB files below that comparison are eligible; missing
files require an existing native archive descriptor. No foreign/parent KDB can
be checkpointed or deleted through an evaluation path.

Closed KDB WALs are checkpointed before initial registration/sealing. The parent
CampaignStorage registers each sample/frozen-input `knowledge.sqlite` as an
`evaluation-kdb` artifact owned by the completed parent attempt. Native Storage
compresses the batch into the parent's existing shared Restic repository, retaining
normal descriptors and deduplicating common database contents. The ordinary
publish-before-finalize path verifies repository bytes and unchanged source
identity before deletion. Retry after descriptor publication can finish source
finalization without another snapshot. A changed source is preserved, not resealed.

Native artifact restore by parent artifact ID rehydrates a KDB for inspection at
the original child path (or a caller-chosen campaign-local destination). State
ledgers, sessions, sources, verifier results and logs remain in place. A completed
archive job does not delete KDBs subsequently restored explicitly for inspection.

Archive workers share the background slot and remaining campaign signal. Native
storage-maintenance locking serializes filesystem work; child process groups are
stopped/reaped on cancellation. Unknown dispatch or surviving descendants remain
fenced rather than being reset. Failures persist for retry even with no new
foreground skill evidence.

Whole-campaign portable export currently cannot preserve background evidence,
evolution and archive queues. Such an export fails explicitly instead of silently
dropping them and claiming completeness. Existing project-scoped exports remain
available, as do ordinary artifact restore and original campaign files. This
limitation does not introduce a second archive or portable package format.
