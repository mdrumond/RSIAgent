# Independent Ascend execution smokes

Run from a complete, dependency-installed source checkout using Node 22.23+.
These are small independent checks, not a campaign or a skill-improvement experiment.

```bash
node examples/ascend-smoke/run.mjs controller PRIVATE_PROFILE_JSON NEW_OUTPUT
RSI_ASCEND_SMOKE_LIVE=1 node examples/ascend-smoke/run.mjs curriculum PRIVATE_PROFILE_JSON NEW_OUTPUT
RSI_ASCEND_SMOKE_LIVE=1 node examples/ascend-smoke/run.mjs actor PRIVATE_PROFILE_JSON NEW_OUTPUT
```

Use a different output directory for every run. Private profiles configure `python`,
`target`, `environment`, `local_execution`, and real `actor`/`curriculum_actor`
provider settings. Credentials remain in ignored configuration. The controller
uses the existing scripted provider fixture and incurs no paid model requests.
The other two commands can spend provider tokens and require the explicit opt-in.

Run A3 from the controller host with the production CPL Ascend environment. Run
A5's entire command on the A5 host with its registered runtime and the production
local Ascend environment. Deployment and observation use the installed CPL client;
project execution on A5 does not transfer through SSH. Required host containment,
provider access and device health must pass independently before launch.
Device runs default to 128 tasks and 4 GiB per local command; explicit limits may
be stricter, never looser. Missing containment fails closed. CPU automated tests
may use their external test guard without making device-execution claims.

- Controller: one fixed project, asymmetric trusted/source paths, original bundle
  removed after import, actual baseline/correctness/performance commands, full Pi
  workflow and positive cleanup. Timings must be positive, with no speedup goal.
- Curriculum: one tiny CPU PyTorch `x + 1` specification, independent reference
  and wrong control, validated then imported. No actor or device execution.
- Actor: one fixed device hello world, normal Pi actor and workflow, a 16-element
  tensor, synchronization and exact arithmetic verification. No benchmark,
  optimization, custom kernel, curriculum generation or model-led summary.

All runs use concurrency one, a tiny isolated production KDB with deterministic
test embeddings, and no background skill A/B. Curriculum has 180 elapsed seconds;
the real actor has 300. Each smoke has an independent 900-second outer deadline.
Controller and actor assert zero curriculum-generation records. Evidence, native
sessions, command receipts and `result.json` remain in the output directory.
Any failed/blocked check must be resolved before a fresh campaign starts.

The generated-project integration smoke uses scripted actual Pi submissions (no
paid curriculum calls), then production generation/import, deletion of original
artifacts, target staging and baseline preflight. Run its portable CPU version:

```sh
node --test --test-concurrency=1 --test-name-pattern 'generated nested starter' tests/ascend/project-generation.mjs
```

For an actual A3/A5 binding, set `RSI_ASCEND_SMOKE_LIVE=1` and
`RSI_GENERATED_SMOKE_PROFILE` to the same ignored private profile used above.
Run on the controller host for A3 and directly on the A5 host. Set
`RSI_TEST_PYTHON` to that host's verified CPU PyTorch interpreter. The profile
must select the canonical Ascend adapter, explicit matching product and mode,
and containment of at most128tasks/4GiB; the same smoke admission checks apply.
Use an outer900-second/4GiB process guard and perform device preflight through
the installed CPL interface. The generated baseline uses only a tiny tensor
on logical `npu:0`; CPU construction/selftests do not establish device timing.
This opt-in test never calls a paid provider or starts an optimization actor.
