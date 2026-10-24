# Moving Ascend campaigns to subjects

New Ascend work uses the ordered subject curriculum through
`run_ascend_subjects.py`, supplied by the prerequisite subject-runtime PR.
The curriculum runs learning projects, qualified targets, and supported optimizer
tasks within one subject list; it no longer requires a separate Phase 2 launch
or a Phase 1-to-Phase 2 handoff.

The following launch paths reject new Phase 2 work before composing a model or
dispatching device execution:

| Entry point | Retired launch actions |
| --- | --- |
| `run_ascend_campaign.py` | `start --phase 2` |
| `run_a3_phase2.py` | `qualify`, `run`, `resume` |
| `run_a5_phase2.py` | `qualify`, `run` |

For a new campaign, create an ordered subject configuration and reuse your
ignored provider/runtime settings with the subject launcher. Use a new campaign
root rather than relabeling an existing Phase 2 checkpoint. See
[the subject curriculum guide](subject-curriculum.md) for configuration, qualified
targets, optimizer bindings, and elapsed-time allocations.

Existing Phase 2 results, artifact identities, and historical controller APIs
remain available for retained-record inspection and compatibility. They are
historical interfaces, not the workflow for starting new training campaigns.
Historical reports do not need provider credentials or device execution:

```bash
python run_a3_phase2.py report --state-root /local/old-a3-campaign
python run_a5_phase2.py report --root /local/old-a5-campaign
python run_ascend_campaign.py status --control /local/old-campaign-control
```

Keep the original roots and their evidence when comparing historical results.
The retirement does not convert or overwrite old checkpoints, reset elapsed
allocations, change host correctness rules, or alter other benchmark lifecycles.
