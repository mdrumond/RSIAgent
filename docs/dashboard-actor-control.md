# Actor feedback runtime

`benchmarks.ascend.actor_control.ActorControl(workspace)` owns one durable inbox.
For graph/target actors use the project's `actor/` directory. For optimizer
actors use the retained optimizer job directory (`jobs/<job_id>/`).

`send(key, text)` publishes ordered user feedback immediately. Repeating the
same key/text returns the original message; conflicting reuse fails. `status()`
returns messages, consumption timestamps, stop intent, and the continuation
checkpoint. `stop(True)` requests a stop at the next safe boundary;
`stop(False)` clears that intent before an explicitly authorized continuation.

Direct, Pi, and optimizer drivers consume instructions before their next model
request. If instructions arrive while a request is in flight, its returned
action is retained as unexecuted and superseded before any tool runs. Existing
device work finishes through its normal retained-operation path. A consumed
receipt means the instruction was persisted in the request context, not that
the provider completed or followed it. Feedback stays outside model-written
compaction summaries.

Graph continuation checkpoints retain the original instruction identity,
conversation/work log, turn, feedback, and editable source. Reopening the same
workspace with another instruction fails: a reviewed restart needs a fresh
linked workspace. Restoring source never restores host verification authority;
the actor must reverify and remeasure before submission. Provider sessions are
recreated; their opaque internal state is not treated as a durable checkpoint.
Existing optimizer loop state retains its source/history and pending operations.

`grant_elapsed(path, key, additional_seconds)` extends a **settled**
`AttemptTimeBudget` allocation idempotently without erasing elapsed usage.
Subsequent budget construction accepts the original allocation plus recorded
grants, preserving the campaign's immutable runtime settings identity. Active
intervals must settle before grants; do not change their timestamps to fabricate
settlement.

These are runtime primitives. The dashboard/campaign owner must serialize
continuations, certify old operations are terminal, archive an existing completed
outcome before reopening it, and avoid changing frozen project/memory identities.
It must reject feedback after publication into accepted memory. Merely sending
to an inbox does not start a worker or invalidate a completed subject cache.
