# Ascend campaigns

Use `run_ascend_subjects.py` for the ordered subject curriculum. It combines
exploration, repair, qualified targets and optimizer tasks in a single workflow.
See [subject curriculum](subject-curriculum.md) for configuration and execution.
A3 uses Ascend C; A5 uses Catlass DSL. Optimizer subjects currently support A3.

The existing `run_ascend_campaign.py` launcher supports Phase 1 campaigns only.
Its `start --phase 1`, `resume`, `continue`, curriculum approval, status,
collection and packaging commands operate on current Phase 1 records. Retired
stage inputs are rejected before controller or provider construction.

Keep campaign and control roots separate, runtime settings ignored, and
source/dependency identities pinned. Use the configured remote-access runtime
and preserve returned execution handles after interruption.

For an original-project retry, provide `--retry-feedback-id`, `--feedback-root`,
`--phase1-root` and `--review`, select one project, and set its additional elapsed
allowance with `--retry-max-seconds`. The retry uses authenticated reviewed
learning, preserves the original acceptance contract and does not ask the
curriculum to replace the task. Feedback selections are retained in
`failure-feedback.json`; observations do not become learned memory without the
current verification and publication process.

New campaigns require fresh roots. Removed stage records and historical
controllers are not supported. Use current subject and optimizer interfaces
for new work rather than relabeling an old run.

Finite campaigns can repeat phases using stable entry IDs:

```json
{"campaign_plan":[{"id":"round1-p1","phase":"phase1","new_projects":4},{"id":"round1-p2","phase":"phase2","new_projects":2},{"id":"round2-p1","phase":"phase1","new_projects":4}]}
```

Assign each project spec a `campaign_entry` matching an entry ID (and, if supplied,
its matching `phase`). Each target counts distinct new original projects from that
entry, including failed originals only after closure, memory delivery, and
postprocessing. Recovery does not contribute to the target. Repeated phases use
new batches; old projects are never replayed. Recoveries retain their original
phase and use the existing scheduler fairness and skill selection. The final
entry ends the campaign without draining queued recovery; the queue remains
available to explicit recovery commands.

The plan and cursor persist in the SQLite control ledger. Started entries are
frozen in full, including their count; completed entries cannot be removed or
edited. Future entries may be changed or appended on continuation. An admitted
project's `campaign_entry` cannot change. A finite plan with too few assigned
projects reports `waiting-inputs`. Existing `phase_sequence`, phase targets, and
single-phase commands remain available when no `campaign_plan` is configured.

Generation/status consumers import `currentCampaignEntry(state, config)` from
`controller/progress.mjs`. It returns `null` without a plan or after its end;
otherwise it returns `{id, phase, new_projects, index, finishedOriginals,
totalOriginals, remainingOriginals, remainingAdmissions}`. `remainingOriginals`
is the unfinished target; `remainingAdmissions` also subtracts active admitted
originals, so generation should fill only that many slots. `campaignProgress`
exposes this object as `currentEntry`. These helpers are read-only projections. Configuration updates validate against
the persisted plan. Entries lock when the controller starts them or admits an
original, so an unstarted first entry remains editable after bootstrap/status.

`campaignProgress().plan` (also exported as `campaignPlanStatus(state, config)`)
is a read-only `{entries, lockedPrefix, currentIndex}` status. Entries include
`index` and `state` (`pending`, `started`, or `completed`). `lockedPrefix` is a
count: disable editing entries with `index < lockedPrefix`. No status read
starts an entry or advances the cursor.
