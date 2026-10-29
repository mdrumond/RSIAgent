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
