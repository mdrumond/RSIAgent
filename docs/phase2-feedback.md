# Phase 2 feedback observations

Optimizer Phase 2 automatically retains outcomes in `CAMPAIGN/feedback` (or
`phase2_feedback_root` in ignored campaign settings). These observations are not
verified reusable lessons. Failures, agent hypotheses, working controls, and
verified-but-slow candidates remain distinguishable. Missing timing is unknown;
speedup below the frozen task's numeric `performance_goal.speedup` (otherwise 1)
is a performance shortfall, not a correctness failure. Existing READY semantics
are unchanged.

Import historical results without running agents or downloading reports:

```sh
python run_ascend_campaign.py feedback-phase2 --feedback-root /data/feedback \
  --source-root /data/old-phase2 --source-root /data/another-phase2
python run_ascend_campaign.py feedback-phase2 --feedback-root /data/feedback \
  --dashboard-config /data/dashboard.json
```

Imports scan retained optimizer states, including nested goal directories, and
deduplicate source-root/attempt identities. Unsupported legacy layouts and missing
evidence are reported. Interrupted jobs are explicitly incomplete observations;
they do not increment completed attempts. Full profiler trees stay remote; only
existing compact result objects and artifact/hash references enter feedback.

`FeedbackStore(root).records()` returns records with `feedback_id`, immutable
`outcome`, and editable `investigation`. `update_investigation(id, **fields)` accepts
status, classification, reason, projects, retries, and evidence. Statuses are open,
investigating, ready-to-retry, resolved, and deferred. Deferral requires a reason;
readiness requires investigation evidence; resolution requires a linked feedback
ID for a successful retry of the identical original task, target, and language.
Importing again never overwrites investigation work or historical outcomes.

This command does not start investigations or restart either phase. Curriculum
integration and explicit bounded retry controls are separate follow-up changes.
