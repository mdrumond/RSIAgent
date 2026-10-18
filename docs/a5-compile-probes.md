# A5 compile-probe budget

A5 Phase 1 agents normally spend one ordinary action for every `{"compile":{}}`
request. This remains the default. A run can instead grant compile requests their own
finite allowance:

```bash
python run_a5_phase1.py run \
  --gate /path/to/gate.json --guide /path/to/agent-guide.md \
  --root /path/to/results --cell-id CELL_ID \
  --tla-root /path/to/tla --profiling-skill-root /path/to/ascend-profiling \
  --catlass-source /remote/catlass --bge-cache /path/to/cache \
  --kdb /path/to/knowledge.sqlite --collection catlass --device 0 \
  --compile-probes 8
```

`--compile-probes LIMIT` requires a positive integer. Each compile request,
including a rejected or failed source, consumes one probe. Requesting probe
`LIMIT + 1` stops the trial with `safety_ceiling`; it does not invoke the compiler.
Ordinary actions continue to use their existing budget. Provider calls, completion
tokens, total turns, and wall time are still measured normally.

The evidence ledger records the ordinary count, probe count, both limits, and whether
each action was exempt from the ordinary budget. Compile is compiler evidence only:
it never launches the kernel, runs host comparison, or establishes correctness. The
agent must still use `run` and finish with the host-owned `submit` verification.

The campaign controller accepts the same option on `reconcile` and forwards it only
to Phase 1 workers. Omit the option to preserve the original metered behavior.
