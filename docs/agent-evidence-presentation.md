# Agent evidence presentation

Adaptive A3 and A5 Actors see compact views of large host observations. Complete
inputs, expected values, observed outputs, compiler diagnostics, measurements and
profiles remain unchanged in the content-addressed artifact store. Large views
include `full_observation_sha256`; nested artifacts retain their existing hashes.
Only original host results decide correctness and publication, never a preview.

Tensor views retain shapes and samples. Execution observations include exact
unequal counts, finite absolute-error maxima and up to eight indexed mismatches.
These are repair clues, not a replacement for the host's comparison tolerance.
Long diagnostics show beginning and end; omitted material remains readable.

Pre-wave memory is a readable lesson preview and complete project-artifact index,
not a base64 copy of every historical memory. Lesson preview text has a shared
8 KiB budget; the index includes every project even when its preview is omitted.
Reviewed Phase 1 seed memories remain visible to adaptive Phase 2. Stored memories
are neither curated nor rewritten. The original snapshot and initial seed also
have artifact hashes.

Actors can inspect any retained local artifact using:

```json
{"read-artifact":{"sha256":"<64-character hash>","offset":0,"limit":8192}}
```

The limit is 1–8192 bytes, default 8192. Offsets are nonnegative bytes. Follow
`next_offset` until null. `text` is convenient UTF-8 with replacement at split
characters; `base64` contains exact page bytes. Reading does not grant evidence
admission, change the current source, or establish host verification. Like other
inspection actions it consumes an Actor action; compile remains unmetered.

Provider/SFT traces record the actual compact messages sent to the model. Raw
host artifacts remain separate and exact for auditing and later training export.
