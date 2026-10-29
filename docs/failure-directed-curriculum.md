# Failure-directed Phase 1

Explicitly investigate retained failure observations without promoting failures
to verified knowledge. Existing campaigns and non-feedback starts are unchanged.

```
python run_ascend_campaign.py start --phase 1 --feedback-root FEEDBACK \
  --failure-id ID --target a3 --root NEW_ROOT --control NEW_CONTROL \
  --objective 'Isolate the observed issue' --project-target 8 \
  --initial-memory MEMORY --config SETTINGS
```

Omitting `--failure-id` selects unresolved observations; successful siblings are
context, not extra tasks. Target/language mismatches are rejected. The selection
is pinned in `failure-feedback.json`. Curriculum sees bounded summaries and can
page authenticated task/source/host artifacts. Profiling report trees stay remote.

The ordinary wave JSON can include `failure_investigations` outside its kernel
grammar. Each probe links a failure, distinct working control, hypothesis and
host-observable outcome. Same-wave controls do not introduce sibling dependencies.
Infrastructure/provider, invalid-specification and suspected unsupported/impossible
issues require explicit classification and reason, never a fabricated kernel lesson.
Budget exhaustion alone does not prove impossibility.

Successful probes remain `investigating`. A subsequent curriculum decision can
recommend `ready-to-retry` only with committed host-evidenced probes and verified
controls. Phase 1 never resolves an original task. Explicit deferrals must explain
why further kernel experiments are inappropriate. Lineage and evidence accumulate
across investigation rounds; immutable failure outcomes never change.

```
python run_ascend_campaign.py start --phase 2 --feedback-root FEEDBACK \
  --retry-feedback-id ID --target a3 --root NEW_RETRY --control NEW_RETRY_CONTROL \
  --objective 'Retry the original target' --project-target 1 \
  --initial-memory UPDATED_MEMORY --phase1-root UPDATED_PHASE1 --review REVIEW \
  --config SETTINGS
```

Retry is a new run with newly reviewed Phase 1 memory. It skips new-goal selection
and Actor case expansion, preserves the original task and expanded frozen input
submission, and checks regenerated suite SHA and comparison policy before Actor
optimization. Original max-cases must match. Missing frozen evidence fails clearly.
This route currently supports A3 executable optimizer tasks; it does not silently
convert legacy A5 tasks. Resolution requires a linked successful original-task
retry; an originally slow kernel must meet its original performance goal.

No import, investigation or retry starts automatically.
