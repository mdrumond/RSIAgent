# Target-driven Phase 2

Phase 1 explores complementary capabilities. Phase 2 starts with a fixed task and
uses actual failures or measured bottlenecks to choose focused practice:

**Attempt target → verify and learn → review → practice → verify and learn → retry target.**

The first attempt always precedes curriculum selection. One practice project runs
at a time. Failed practice attempts consume the same budget as successful ones.
After two practice attempts, a target retry is mandatory. The default permits
three target attempts. Passing practice never establishes target readiness, and
practice after a passing target still requires a fresh target attempt.

Correctness uses the existing deterministic inputs, numerical oracle, and
quantitative tolerances. Device runtime is an optimization objective, without an
invented speedup threshold. Debug, print, sanitizer, and profiler runs stay
separate from ordinary timing. Unavailable tools or timing are not zero-cost
measurements.

## Qualify an achievable goal

The host must execute a working reference on the selected runtime before enabling
a target. A reference that passes on another device, dtype, shape, or toolchain
does not qualify this target. Use the same composed `ProjectExecutor`, selected
runtime identity, and target policy as the production experiment. Existing native
or Catlass project backends implement that executor interface.

```python
import json
from pathlib import Path
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
from benchmarks.ascend.phase2_targets import qualify_target

# executor and target_policy come from the selected native/Catlass composition.
# runtime_identity records observed target, runtime, device, toolchain, product,
# and compiler identities from that same selected runtime.
spec = ProjectSpec.from_mapping(json.loads(Path("copy-spec.json").read_text()))
store = ContentAddressedArtifacts(Path("private-reference-artifacts"))
goal = qualify_target(spec, Path("working-copy-source.txt").read_text(),
                      runtime_identity, executor, store, policy=target_policy)
Path("copy-goal.json").write_text(json.dumps(goal.as_dict(), indent=2))
```

Start with a verified movement, vector fusion, reduction, or matrix reference;
each concrete spec requires its own qualification. Optional `timings=` accepts
positive device duration records with unit (`ns`, `us`, `ms`), measurement
conditions, and a retained `evidence_sha256` in the same store. These are reference
observations, never a mandatory speedup. Keep development goals separate from
held-out evaluation goals.

Reference source stays private. The curriculum receives the fixed specification,
latest host verdict, bounded compiler/runtime diagnostics, numerical mismatch
summary, observed host facts, prior lessons, remaining budgets, and configured
available tools. It must name the observed weakness, controlled change, expected
evidence, and how practice informs the unchanged target.

## Start and resume

Add these keys to the ignored existing campaign settings JSON alongside its
`model`, `source_manifest`, and `phase1_arguments`:

```json
{
  "target_artifact_store": "/absolute/path/to/private-reference-artifacts",
  "target_runtime": {
    "target": "a3",
    "runtime": "registered-runtime-name",
    "device": "selected-device",
    "toolchain": "observed-toolchain-version",
    "product": "observed-product",
    "compiler": "observed-compiler-revision"
  },
  "target_available_tools": []
}
```

Use observed runtime values, not the illustrative strings above. Only list tools
that the selected backend can actually execute. A5 uses the same workflow with an
A5-qualified contract and A5 configuration.

```bash
python run_ascend_campaign.py start \
  --control /absolute/path/to/private-control \
  --root /absolute/path/to/phase2-campaign --target a3 --phase 2 \
  --objective 'Repair and optimize the fixed copy target' \
  --config /absolute/path/to/settings.json \
  --initial-memory /absolute/path/to/phase1-initial-memory.bin \
  --phase1-root /absolute/path/to/reviewed-phase1 \
  --review /absolute/path/to/phase1-review.json \
  --target-goal /absolute/path/to/copy-goal.json --project-target 1 \
  --target-attempt-budget 3 --practice-attempt-budget 2
python run_ascend_campaign.py status --control /absolute/path/to/private-control
python run_ascend_campaign.py resume --control /absolute/path/to/private-control
```

The goal and all reference objects are pinned privately in the control directory
at start. Keep that directory with the campaign for execution recovery; the
public campaign archive does not distribute the private reference source.
Existing campaigns without a target goal continue using their original Phase 2
behavior on resume. New Phase 2 starts require a goal. Phase 1 and direct legacy
entrypoints retain their existing semantics.

`READY` requires a freshly host-verified target and a curriculum decision that no
further useful practice is needed. `STALLED` and `BUDGET_EXHAUSTED` are incomplete
outcomes. Infrastructure exceptions leave the exact pending branch and its
retained handles for explicit resume; they do not consume a semantic attempt.
Target-driven budgets and identities cannot be changed in place. Wave approval
mode is currently unsupported for target-driven campaigns and is rejected before
creation.
