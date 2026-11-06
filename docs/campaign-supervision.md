# Campaign supervision

Run supervision only after the local gate and both targets' baseline qualification
receipts pass. The instance must have a fresh `campaign_cohort`, finite native
campaign plan/allocation and `campaign_source_revision` matching the qualification
document's expected source revision. All machine/provider settings remain in the
private instance profile.

```bash
node INSTANCE/machinery/cli/campaign-supervisor.mjs --instance INSTANCE \
  --cohort COHORT --qualification PRIVATE_QUALIFICATION_JSON --revision RELEASE_SHA
```

The module API is `superviseCampaign({instance, cohort, qualification,
sourceRevision, signal})`. Its default CLI and runtime come from that instance's
installed machinery. The public CLI owns detached controller requests and their
durable receipts; supervision does not introduce another execution backend.

One PID/start/boot claim prevents competing supervisors. Status is checked every
minute and stopped-controller ownership is reconciled every five minutes. A live
request or local worker is observed rather than duplicated. After local ownership
is positively closed, native `resume` observes retained remote handles and preserves
their reservations, attempts, sessions and campaign budget cycle. Supervision
never clears execution fences or grants a new elapsed allocation.

Initial launch requires 64 GiB free. Below 16 GiB during a campaign, supervision
requests pause/checkpoint and reports attention needed. It does not delete files.
Unknown ownership or repeated infrastructure failures receive thirty elapsed
minutes before attention is reported; legitimate active workers are not treated
as stalled merely because they have long actor allocations.
The infrastructure wait uses elapsed monotonic time and retains its checkpoint
across supervisor restarts; restarting supervision does not refill that window.

SIGTERM/SIGINT, operator pause, finite completion and elapsed exhaustion request a
checkpoint. Supervision waits through pause grace plus sixty seconds for closure;
unproven closure is reported as `checkpointPending`, never forced. Completion does
not drain additional A/B work. Ignored instance-local `.supervisor/status.json`,
`state.json` and `events.jsonl` retain compact lifecycle reports without provider
arguments or configuration. Packing, destructive cleanup, baseline replacement
and machinery upgrades require the supervisor to stop first. Compression of
already closed attempt artifacts remains available during supervision.
