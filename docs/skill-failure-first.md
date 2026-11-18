# Failure-first skill evaluation

The initial rollout evaluates correctness guidance on A3. Optimization findings
remain deferred, and harness findings require human review. Foreground campaign
work must be paused while this worker uses four healthy, idle, unleased devices.

Start the worker with `node cli/main.mjs skills-ab --instance INSTANCE`.
`--once` processes one family. The existing `status` command includes its stage,
screen results, retained resource bindings, and stopping reason.

Configuration must explicitly enable `skills.failureFirst.enabled`, enable
`skills.background`, clear `skills.tuningDisabled`, and pin the installed source
commit in `skills.failureFirst.sourceRevision`. The original campaign elapsed
allocation is required; starting this worker does not renew it. Foreground
`wave` and `resume` reject execution while this worker is enabled.

Each ranked correctness family follows these gates:

1. Recover the failed actor source and linked negative verification. Missing
   evidence advances the queue without inventing a replacement failure.
2. Isolate a development task and two structurally different transfer tasks.
   Qualify passing references and negative controls before model trials.
3. Run five baseline attempts. At least three must reproduce the same failure
   mechanism before any instruction is written.
4. Derive three short, general instructions using successful searchable memory.
   Review their evidence and applicability before freezing the complete skills.
5. Evaluate each candidate five times alongside five fresh development baselines.
   Choose the winner using development results only.
6. Run five fresh baseline and five winner attempts on each transfer task. Each
   transfer must gain at least two full successes, with no control regression.
7. Recheck the stored evidence and publish exactly the reviewed, tested skill
   version through the existing atomic publisher.

An experiment has two elapsed hours, including isolation, qualification, writing,
review and evaluation. Isolation is bounded to thirty minutes. Trials receive
180 seconds for the actor and 300 seconds for independent verification, with at
most four trials in flight. The model context cap is 200,000 tokens. Pauses and
resumes retain elapsed spend, frozen evidence, guidance, and device bindings.

Memory uses a generation-keyed private snapshot. Before timed actors start, a
cold retrieval must finish within 240 seconds and four distinct warmed searches
must each finish in under ten seconds. Private oracles and other actors' work
are hidden from the native Pi actor filesystem.

A qualified scientific rejection advances the queue. Infrastructure failure,
malformed model output, unknown execution custody, or exhausted experiment time
stops the worker and disables A/B while preserving evidence. Inspect the recorded
reason and retained handles before explicitly restarting; the worker does not
repair a failed live campaign in place.
