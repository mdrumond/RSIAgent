# Campaign artifact collector

Run the collector as its own process in a registered durable Screen session,
using the supported remote-access session interface. It performs no kernel work
and never writes campaign memory, changes campaign outcomes, or deletes remote
reports. Campaign launch and completion must not wait for this worker.

```sh
python -m benchmarks.ascend.artifact_collector CAMPAIGN_ROOT \
  --artifact-store ARTIFACT_STORE --state QUEUE_STATE
```

The campaign root must already exist. Both output locations must be outside it.
`--once` advances one transfer for integrations; otherwise the worker checks its
queue every 30 seconds and discovers new closed report descriptors every 300
seconds. The Python interface is `ArtifactCollector(...).collect_once()` or
`.watch(stop=threading.Event())`. Status includes counts, the next pending retry
time, and whether low disk space pauses transfers. The default free space floor
is 5 GiB; `--free-floor-bytes` changes it.

The queue discovers historical and current raw profiling directories from the
training annex inventory. Duplicate target/path/digest identities share one
entry. Downloads use only the installed user-wide `cpl-remote --json` interface,
with stderr retained separately. Each transfer writes to a stable partial tree;
the annex digest must match before atomic publication as `ARTIFACT_STORE/ID/`.
Queue state, partial data, and the latest transport response survive restarts.
Completed trees that fail verification are retained under `.corrupt-*` names,
allowing the next attempt to start with a clean tree.
There is one transfer at a time, including across overlapping worker launches.
Worker attempts back off 120 seconds; after three failures they defer 15 minutes.
The registered transport may also apply its own native retry policy.

Remote-agent handles are preserved across uncertain observations; completed
handles are materialized using `fetch`, without submitting a replacement.
The transfer subprocess inherits the worker lock, so a replacement worker waits
until an orphaned subprocess exits. An interrupted SSH transfer then resumes
the same native partial tree. A remote-agent submission that produces no usable
handle remains `uncertain` for operator investigation: retrying it could create
a duplicate. Keep the queue and its adjacent transport JSON/stderr journals for
that investigation. No automatic reset or deletion of uncertain work occurs.

These mechanisms isolate failures and resume partial downloads; they do not
claim a bandwidth cap. Configure transport/runtime routing in the ignored CPL
registry rather than in campaign records or collector arguments.
