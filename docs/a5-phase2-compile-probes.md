# A5 Phase 2 compile probes

Phase 2 agents can optionally ask the checked BZ-A5 host to compile a complete
candidate before submitting it for numerical verification. Probes are disabled
by default. Enable a finite allowance for each semantic attempt with:

```bash
python run_a5_phase2.py run \
  --root "$PHASE2_ROOT" \
  --snapshot "$PHASE2_ROOT/cells/$CELL_ID/snapshot" \
  --cell-id "$CELL_ID" \
  --phase1-root "$PHASE1_ROOT" --gate "$GATE" --guide "$GUIDE" \
  --phase2-guide "$PHASE2_GUIDE" \
  --phase2-cpl-skills-revision "$CPL_SKILLS_REVISION" \
  --tla-root "$TLA_ROOT" --catlass-source "$CATLASS_SOURCE" \
  --device 0 --env-file "$ENV_FILE" --compile-probes 3
```

The campaign controller accepts the same `--compile-probes N` option and
forwards it to Phase 2 qualification and run workers.

An agent requests a probe with exactly one JSON action:

```json
{"compile_probe":{"source":"complete Python candidate source"}}
```

A valid probe does not consume the ordinary provider-turn allowance or a
semantic attempt. Actual provider token use is still retained. The independent
probe allowance prevents an unbounded loop.

The host response identifies the source digest, bound case, compiler exit code,
bounded diagnostic, and retained session. It explicitly says that the action
was compilation-only and established no correctness. The checked driver exits
immediately after `tla.compile`; it never launches the artifact, compares
outputs, profiles it, or admits the candidate. Only the normal host verifier
can establish correctness.

Probe source and result records live below the attempt's
`compile-probes/probe-NN/` directory. Repeated processing of the same retained
probe reuses its digest-bound result. Transport and observation failures remain
infrastructure exceptions and do not create semantic compiler evidence.
