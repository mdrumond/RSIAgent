# Automatic failure remediation

Enable `automatic_remediation: true` in the ignored campaign settings used by
`run_ascend_campaign.py start --phase 1`. The existing campaign time allocation
also bounds remediation. `failure_active_seconds` defaults to 1800 seconds per
failure. The original project target and specifications remain unchanged;
remediation may continue beyond that target while an eligible failure remains.

Optional settings:

```json
{
  "automatic_remediation": true,
  "failure_active_seconds": 1800,
  "feedback_root": "/configured/feedback",
  "feedback_sources": [
    {"type": "phase1", "root": "/retained/phase1"},
    {"type": "phase2", "root": "/retained/phase2"},
    {"type": "optimizer", "root": "/retained/jobs"}
  ],
  "optimizer_worker_config": "/configured/optimizer.json"
}
```

The default feedback store is the Phase 1 campaign's `feedback` directory.
Standalone optimizer workers also import terminal jobs into their configured
`feedback_root`, defaulting to the jobs directory's `feedback` child. Imports
are idempotent and never execute work. Manual imports accept each source type:

```bash
python run_ascend_campaign.py feedback --source-type optimizer \
  --source-root /retained/jobs --feedback-root /configured/feedback
```

Curriculum receives unresolved observations after every committed wave. It may
mix new projects with linked remedial probes and controls. Retained host evidence
supports a ready-to-retry recommendation, which triggers an isolated execution
of the unchanged original specification. Successful remedial projects alone never
mark an original failure fixed. Failed original retries retain their evidence
and return the original observation to failed for further investigation.

A3 external optimizer/Phase 2 retries require `optimizer_worker_config` and
retained frozen suite/comparison-policy evidence. They use authenticated committed
Phase 1 memory, explicitly labeled `committed-phase1`; this does not manufacture
a human review or change the existing reviewed handoff. The configured worker's
execution mode, including local execution inside a registered runtime, is retained.

`remediation.json` records per-failure budgets and original retry roots. Original
retries live under `original-retries/<feedback-id>/retry-N`, retain the original
project ID, and are excluded from creating duplicate remediation obligations.
Budget exhaustion leaves the failure unresolved with its history intact.
