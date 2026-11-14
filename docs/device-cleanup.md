# Background Ascend device cleanup

Ascend commands distinguish payload termination, output availability and device
fence release. A known terminal payload returns without waiting for host cleanup.
If its matching admission receipt already proves release, the reservation closes
immediately. Otherwise `cleanupPending` identifies a durable token-scoped job;
the old physical device remains reserved while subsequent commands can use other
healthy enabled devices. An unknown payload never enters this queue as closed.

The default is asynchronous. Set environment option `asyncCleanup:false` for the
legacy synchronous protocol. A single owned Node coordinator per user lease
directory services at most two cleanup jobs concurrently through the existing
Ascend environment and CPL client. It creates no custom remote execution tool.
The submitting actor may exit; cleanup custody, native journals and PID/start/boot
identity survive in the queue and token-specific pointers. A resumed environment
reclaims only a positively dead worker claim and observes the original identities.
Old cleanup cannot unlink a newer command's active lease pointer.

Each job retains the existing `probeObservationSeconds` cleanup allocation from
its original enqueue time. Queue wait consumes that allocation. Retry and resume
never refill it. Failed or expired cleanup leaves its reservation fenced and its
status inspectable; it does not manufacture closure. Ordinary admission cannot
silently grant an expired cleanup another allowance. Explicit administrative
exact-token recovery remains separate.

Queue state and logs live under the configured user lease directory's `.cleanup`
directory. Campaign `device-cleanup-service.json` records the coordinator identity
for memory accounting, including after the actor exits. Active cleanup writers
for that campaign prevent a stopped-instance upgrade; another campaign's jobs or
an expired fence without a live writer do not. A stopped writer does not prove
that its physical fence is free.

Cleanup events use their own journal rather than adding unresolved cleanup jobs
to the actor's execution history. Thus completed payloads can settle and release
scheduler host slots while their physical reservation remains protected. Legacy
unknown histories still require their original positive closure evidence.

This change moves device-fence cleanup off the actor path. The existing inline
terminal-result protocol already avoids transfers for small outputs; large/binary
output collection retains its existing synchronous fallback. Missing output is
not a correctness pass, and this queue does not invent a native ID for a lost
historical artifact transfer.
