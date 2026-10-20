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
deduplicate retained optimizer job/attempt identities. Restored campaigns retain
their existing feedback IDs and investigation state, including older path-keyed
records. Historical records without a retained job ID fall back to source-root
identity and explicitly report that relocation deduplication is unavailable.
Unsupported legacy layouts and missing
evidence are reported. Interrupted jobs are explicitly incomplete observations;
they do not increment completed attempts. Full profiler trees stay remote; only
existing compact result objects and artifact/hash references enter feedback.
Target identity comes from campaign metadata or explicit `ascend-a3`/`ascend-a5`
seed identities, never from a transport host label. Missing identity stays unknown.

`FeedbackStore(root).records()` returns records with `feedback_id`, immutable
`outcome`, and editable `investigation`. `update_investigation(id, **fields)` accepts
status, classification, reason, projects, retries, and evidence. Statuses are open,
investigating, ready-to-retry, resolved, and deferred. Deferral requires a reason;
readiness requires investigation evidence; resolution requires a linked feedback
ID for a host-verified retry of the identical original task, target, and language.
A performance shortfall additionally requires meeting the original numeric goal.
Correctness-only retries do not require timing. Verified controls without timing
are retained as correctness successes, not unresolved failures.
Importing again never overwrites investigation work or historical outcomes.
Changed source outcomes produce a warning instead of overwriting history.
`read_evidence(digest, offset=0, limit=8192)` authenticates a linked local artifact
and returns a bounded text page with byte `next_offset` (or null at EOF). It never
fetches remote reports or embeds entire profiler trees.
Campaign packages retain the default `feedback/` tree and Phase 1's
`phase2-feedback.json` and `failure-investigations.json`, including investigation
state and retry links. Original absolute evidence locations remain provenance;
packaging does not make missing remote artifacts locally available. Unknown
target or language identity cannot establish retry-based resolution.
Resolution also authenticates both retained baseline artifacts and requires the
same frozen expanded-suite hash and comparison policy. Missing or corrupt baseline
evidence leaves the observation unresolved; it does not discard the observation.
Retry IDs must refer to observations already retained in the same feedback store.

This command does not start investigations or restart either phase. Curriculum
integration and explicit bounded retry controls are separate follow-up changes.
