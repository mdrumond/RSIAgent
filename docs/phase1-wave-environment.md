# Native Phase 1 environment admission

`build_native_adaptive_phase1` checks each configured worker before selecting a
new wave or executing an approved wave after resume. The check compiles the
host-owned three-input FP32 probe, loads the resulting library, runs it on the
worker's selected device, and compares all 32 output values with the exact host
oracle. It uses the same transport, architecture, compiler and runtime path as
project execution. Workers are checked serially.

A failed compile, shell setup error, failed execution or incorrect output blocks
curriculum and Actor work. Probe time is charged to the existing campaign elapsed
allocation. Checks use fresh attempt identities on every live invocation, so a
successful old compilation cannot qualify a changed environment on resume.

Reports live under `environment-checks/wave-NNN/ATTEMPT/report.json`, alongside
probe artifacts. Remote transport journals retain their original job handles;
observe those handles when the outcome is unavailable. A failed check does not
reset allocations, retry commands or publish project memory. Fix the selected
environment before explicitly resuming within the remaining campaign allocation.
Read-only reports and retained-wave finalization do not execute probes. Reviewed
Phase 2 and A5 compositions keep their existing admission behavior.
