# Kernel optimizer jobs

The standalone `optimizer` package retains submissions, memory snapshots, events,
artifacts, budgets, and remote handles independently of the dashboard process.
These commands create and inspect jobs; the execution worker is a separate layer.

```sh
python -m optimizer discover --memory-root /configured/completed-phase1
python -m optimizer create --store /configured/optimizer-jobs \
  --memory-root /configured/completed-phase1 --source reference.py \
  --entrypoint forward --signature signatures.json --idempotency-key demo-1
python -m optimizer inspect --store /configured/optimizer-jobs JOB_ID
python -m optimizer events --store /configured/optimizer-jobs JOB_ID
python -m optimizer stop --store /configured/optimizer-jobs JOB_ID
python -m optimizer resume --store /configured/optimizer-jobs JOB_ID \
  --additional-attempts 5 --additional-seconds 600
```

`signatures.json` contains `{"signatures":[{"args":[{"shape":[257],"dtype":"float32"}],"kwargs":{}}]}`.
Source and input execution/validation belong to the reference runner. JobStore
preserves them exactly without executing submitted code.

Discovery examines only supplied campaign roots. If the reviewed seed lives
separately, pass its configured parent using repeatable `--snapshot-root`; that
root must contain `cells/*/snapshot/snapshot.json`. Discovery explicitly reports
legacy, incomplete, corrupt, pending, A5, and missing-reviewed-seed sources as
unavailable. A READY string alone never admits memory. Native target-driven
sources require READY plus the final published verified target; reviewed adaptive
sources require their successful-project target or SATURATED with published memory.

Snapshots freeze reviewed or committed Phase 1 memory, publication chain,
source identity, and referenced content-addressed artifacts as portable bytes with
SHA-256 identities. Full profiler reports are not needed to consume memory.

Defaults are 25 evaluated candidate revisions and 7200 seconds, whichever limit
is reached first. The worker must account active wall time through `account_time`,
record evaluated revisions through `record_attempt`, and retain a returned remote
handle before observation. A stop request on a running job must be observed by the
worker, which owns cancellation and terminal acknowledgement. Queued stops take
effect immediately. Resume requires explicit positive added budgets and retains
all prior handles/evidence. Handles are never cleared or implicitly resubmitted.

Per-job locks serialize mutations. Hashed events are the durable commit boundary;
manifest reads recover a committed update if interruption precedes its cached
manifest replacement. Artifacts and immutable submissions are checked on read.
