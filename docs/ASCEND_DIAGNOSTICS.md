# Ascend diagnostic actions

Adaptive Phase 1 Actors can request `{"request-debug":{}}` or
`{"request-sanitizer":{"tool":"memcheck"}}`. The sanitizer tools are
`memcheck`, `racecheck`, `initcheck`, and `synccheck`. Use memcheck before
investigating races or synchronization. Write a candidate first; host
verification is deliberately not a prerequisite for diagnosing a failed
candidate. Submission still requires the normal exact-source host oracle.

Production binds these actions to the existing checked BZ-A5 project executor.
The same content-addressed kernel, manifest, materialized inputs, host driver,
and frozen runtime provenance are staged. A retained operation handle and raw
command transcripts are returned as diagnostic evidence. Actor input cannot
select executable paths, flags, debugger commands, environment, or timeouts.

msdebug runs a fixed LLDB batch script with user initialization disabled:
pending kernel-name regex breakpoint, run, `ascend info cores`, and backtrace.
A breakpoint may remain unresolved for Catlass JIT code; inspect the transcript
for an actual device stop. The installed tool must support these LLDB options
(verified with BZ-A5 msdebug 26.2.0 `--help`). The harness never enables the
system driver debug channel. Missing executable, missing symbols, permission
failures, and unsupported code are observations, not successful debug claims.

mssanitizer runs the selected tool around the exact host replay, with PyTorch
NPU memory caching disabled. Its report can include framework setup kernels.
No clean result or kernel instruction coverage is inferred from process exit
zero. Every result explicitly reports coverage as unconfirmed and cannot
produce HOST_VERIFIED, timing, or profiling evidence. The Actor must inspect
reported kernel names, errors, and tool limitations before drawing conclusions.

Each tool is bounded to 120 seconds and 48 KB of captured transcript, with
truncation explicitly recorded. Timeout interrupts, then kills the tool process
group, including the inferior; this does not guarantee device recovery from an
NPU hang. Do not intentionally launch hanging variants on a shared device.

Do not apply older AscendC sanitizer compilation flags to Catlass automatically.
The official sanitizer documentation states Ascend 950 does not support
`--cce-enable-sanitizer`, `--sanitizer`, or `-O0`. This implementation does not
claim compiler instrumentation support or source-level breakpoint coverage.
Future compiler-specific integration needs separate device evidence.

Official references:

- [msdebug user guide](https://github.com/Ascend/msdebug/blob/master/docs/en/user_guide/msdebug_user_guide.md)
- [mssanitizer user guide](https://github.com/Ascend/mssanitizer/blob/master/docs/en/user_guide/mssanitizer_user_guide.md)
- [Sanitizer compiler options and Ascend 950 restrictions](https://github.com/Ascend/mssanitizer/blob/master/docs/en/user_guide/compile_option_config.md)
- [Inherited LLDB batch options](https://lldb.llvm.org/man/lldb.html)
