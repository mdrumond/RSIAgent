# Native Phase1 through CPL

`benchmarks.ascend.cpl_native` adds a transport for native A3 campaigns without
changing historical candidate or measurement adapters. Inject
`CplNativeCandidateBackend` into the shared native production builder and use
`CplNativeMeasurements` as the measurement factory. The candidate contract stays
`compile(plan, local_directory)`, `execute_receipt(compilation)`, and
`remote_candidate_directory(plan)`; measurements retain `timing` and `capture`.

Configure the skill-owned `cpl_remote` executable, registry `target`, named
`runtime`, an existing absolute `remote_workspace`, `physical_device`, and local
`state_directory`. `profile` retains the plan's historical semantic binding;
it does not select a transport route. Pass a `remaining_seconds` callable to cap
nested work by the campaign's remaining elapsed allocation. The selected named
runtime supplies compiler/CANN/Python activation; no environment activation paths
or SSH routes belong in campaign source. This binding currently uses absolute
upload paths and working directories supported by SSH registry targets.

The caller must preflight capabilities, discover actual product/tool versions,
and select a healthy device through CPL before NPU work. Ordinary execution
masks the selected physical device to logical zero. Timing requires healthy idle
isolation and a separate ordinary timing replay. Profiling reuses the native
A3 driver and raw exact-kernel report semantics; availability is distinct from
host correctness and no A5 metric interpretation is applied.

Every upload/run is recorded before submission. Once a job handle is known,
resume observes that same `remote:target:job:id`; completed stdout, stderr and
exit status are retained locally. A lost submission response without a handle
is reported as ambiguous and is never automatically resubmitted. Inspect the
retained operation and use the skill's observation/recovery interface before
continuing. Completed source/library identities remain bound to the immutable
candidate, selected route/runtime, and physical device. Timing verifies values
against the host-owned reference and retains samples, environment, compact raw
evidence and the full CPL job handle. Vendor profiler reports remain remote.
