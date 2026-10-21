<p align="center">
  <img src="docs/assets/rsiagent-icon.svg" alt="RSIAgent icon" width="104" height="104">
</p>

<h1 align="center">
  <img src="docs/assets/rsiagent-wordmark.svg" alt="RSIAgent" width="480">
</h1>

<h2 align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="docs/assets/rsiagent-subtitle-mobile.svg">
    <img src="docs/assets/rsiagent-subtitle.svg" alt="Autonomous Exploration for Recursive Self-improvement in New Environments" width="100%">
  </picture>
</h2>

<p align="center">
  <a href="https://arxiv.org/pdf/2609.15364"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat" alt="Paper: arXiv"></a>
  <a href="https://huggingface.co/papers/2609.15364" title="No. 6 on the September 15, 2026 Daily Papers list; checked September 16, 2026 (UTC)"><img src="https://img.shields.io/badge/HF_Daily_Papers-%236_%C2%B7_2026--09--15-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Daily Papers: #6 on the September 15, 2026 list"></a>
  <a href="https://aetherlabsai.github.io/RSIAgent/"><img src="https://img.shields.io/badge/Website-RSIAgent-6554c0?style=flat" alt="Website: RSIAgent"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat" alt="License: Apache 2.0"></a>
</p>

<!-- Rank snapshot verified September 16, 2026 (UTC): https://huggingface.co/papers/date/2026-09-15 -->

<p align="center">
  <a href="#news">News</a> · <a href="#demos">Demos</a> · <a href="#method">Method</a> · <a href="#results">Results</a> · <a href="#installation">Quickstart</a> · <a href="#ascend">Ascend campaigns</a> · <a href="#dashboard">Dashboard</a> · <a href="#documentation">Documentation</a> · <a href="#citation">Citation</a>
</p>

https://github.com/user-attachments/assets/11cb3919-1073-4ba3-8bcd-37a15ec53c33

<a id="news"></a>

## 📰 News & Highlights

> **🔥 Headline feature in 机器之心 · September 2026**
>
> **Open-source models surpass GPT-6 Astra on challenging computer-use benchmarks.**
> Discover how RSIAgent turns autonomous exploration into reusable experience—without updating model weights.
>
> [![WeChat article reads: 40K+](https://img.shields.io/badge/WeChat_reads-40K%2B-2E7D32?style=flat-square&logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/bg0tkvHsKTZ88uBNKUdVTQ)
>
> <sub>Read-count snapshot: September 16, 2026.</sub>

<a id="overview"></a>

## 🧭 Overview

RSIAgent is a **training-free framework for recursive self-improvement** in new
digital environments. It coordinates the Curriculum Agent, Actor Agent, and
Verifier Agent to discover how an environment works, check what they learn against actual execution,
and retain reusable knowledge in persistent memory. Model parameters stay fixed
throughout exploration and downstream task execution.

The paper's central strategy is **broad-then-deep exploration**: first acquire
diverse experience, then investigate hard cases, hidden constraints, and boundary
conditions. The resulting memory contains procedures, scripts, and failure lessons
that the Actor Agent reuses for downstream task execution.

<a id="demos"></a>

## 🎬 Demos

https://github.com/user-attachments/assets/b939a9c2-4acb-4c30-b72e-18bab646529c

<a id="framework"></a>

## 🧩 Framework

[![RSIAgent framework: parallel Broad Recursive Self-exploration, sequential Deep Recursive Self-exploration, and test-time reuse of frozen memory, illustrated with FreeCAD.](docs/assets/framework.png)](docs/assets/framework.png)

*The paper's original framework figure, illustrated with a FreeCAD task. Broad
experience is progressively refined into targeted memory, then frozen for reuse.
Click the figure for full resolution. [Figure provenance](docs/PAPER.md#figure-provenance).*

<a id="method"></a>

## 🔄 Method

Three agents carry out the recursive learning loop:

- **Curriculum Agent** chooses informative exploration tasks using prior outcomes and
  accumulated knowledge, then decides whether further practice is useful.
- **Actor Agent** interacts with software through executable Python or Bash programs
  and visual observations. After verification, the same Actor Agent distills its
  experience and reconciles it with existing memory.
- **Verifier Agent** independently inspects task requirements and the resulting
  environment. Its feedback grounds learning; it cannot read the Actor Agent's
  private reasoning or memory.

The paper has **two exploration stages followed by test-time memory reuse**.
**RSI and test-time execution use the same agent framework**, with fixed model
parameters throughout. At test time, memory is frozen, and the Curriculum Agent
and memory updates are disabled. The implementation exposes these as three
runtime phases:

| Runtime phase | Paper stage | Learning and execution |
| --- | --- | --- |
| **Phase 1** | **Broad Recursive Self-exploration (BRS)** | The Curriculum Agent proposes diverse projects. Actor Agents execute and Verifier Agents check them in parallel from a shared starting memory. After the complete wave, Actor Agents consolidate their experiences in order. |
| **Phase 2** | **Deep Recursive Self-exploration (DRS)** | Target attempts reveal gaps and fragile successes. The Curriculum Agent selects focused practice; each verified experience updates memory before subsequent practice or another target attempt. |
| **Phase 3** | **Test-time memory reuse** | The Actor Agent uses frozen memory to guide task execution, interacting with the Verifier Agent through the same action–verification loop used during RSI. Sealed official evaluation follows task execution and verification. |

Memory is the persistent learning state across tasks. Interaction histories and
task environments are reset between independent attempts; the Agent framework
remains unchanged. Both grounded successes and failures can teach useful lessons.
Official benchmark scores are kept outside the learning loop. See
[Architecture](docs/ARCHITECTURE.md) for the role interfaces, wave memory barrier,
and stopping rules.

<a id="results"></a>

## 📊 Results

The manuscript reports these **mean partial-credit scores (%)** for the shared
harness coordinating the Actor Agent and Verifier Agent, with and without RSI:

| Benchmark and reporting coverage | RSIAgent w/o RSI | RSIAgent |
| --- | ---: | ---: |
| OSWorld 2.0 · 0808 offline · 82 tasks | 71.97 | **78.98** |
| Agents' Last Exam · Near-term · 67 tasks | 83.75 | **84.82** |

These are the manuscript's reported aggregates. The RSI column uses 41 recorded
RSI entries for OSWorld and 19 for ALE, retaining baseline scores for the other
41 OSWorld tasks and 48 ALE tasks. ALE includes all 67 Near-term tasks, including
the three GPU baseline results. The RSI column includes selected retries and
checkpoints with differing budgets; it is not an average over matched repeated
runs. ALE also includes qualified local
regrades and protocol variants. See the [paper and reporting notes](docs/PAPER.md)
for the full scope, full-credit metrics, and aggregation details.

The paper also examines stage ablations, memory growth, and game development.
Its failure analysis identifies three limits to improvement: practice can miss
the relevant weakness, verification can accept incomplete work, and memory can
preserve an incorrect rule. The quality of exploration, verification, and memory
consolidation therefore matters alongside the amount of practice.

<a id="benchmarks"></a>

## 🧪 Benchmarks

| Integration | Pinned release | Public batch |
| --- | --- | --- |
| OSWorld-V2 | August 8, 2026 | 108 tasks, Docker/QEMU |
| Agents' Last Exam (ALE) | `d10fb61a14f9719774c3520c5763068b28ef5546` | 67 Near-term tasks; 64 CPU tasks and 3 completed GPU-task baselines |

All three GPU-task baselines are complete. Chroma Key ran on a Google Cloud
(GCP) VM using the official ALE Windows image and an NVIDIA L4 vWS GPU.

Only the current runtime is included. Both integrations use the same Actor,
Verifier, Curriculum, and memory protocol. Benchmark setup and grading remain
outside the learning process.

<a id="repository-layout"></a>

## 🗂️ Repository layout

```text
run_osworld.py        OSWorld batch entrypoint
run_ale.py            ALE preparation, execution, and reporting
run_ascend_campaign.py Explicit Ascend phase launcher and artifact collection
benchmarks/
  osworld/           OSWorld stages, VM adapter, and evaluation
  ale/               ALE host, worker, and VM adapters
  ascend/            Shared A3/A5 campaigns, feedback, and training annexes
  a5kernels/         Catlass DSL and adaptive kernel-learning integration
dashboard/           Campaign viewer and optional A3 launch controls
optimizer/           Executable kernel submissions and native A3 workers
core/                Shared Actor and Verifier runtime
explore/             Curriculum, learning, memory, and recovery
env/                 Shared guest transport and isolation
llm/                 Model clients
config/              Role profiles and benchmark configurations
scripts/             Setup and individual-study helpers
tools/               Preparation, smoke checks, and recovery utilities
tests/               Regression tests
docs/                Guides, architecture, and attribution
```

The root entrypoints are the starting point for runs. Internal
OSWorld stages are Python modules under `benchmarks/osworld/`; see the
[source map](docs/ARCHITECTURE.md#source-map) for their responsibilities.

<a id="installation"></a>

## ⚙️ Installation

For OSWorld and ALE, use Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a Linux host with Docker and access to `/dev/kvm`. Start with the repository
and your own model API credential:

```bash
git clone https://github.com/AetherLabsAI/RSIAgent.git
cd RSIAgent
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY` in `.env`. Paths default to this checkout and sibling
benchmark directories. Export `RSIAGENT_ROOT`, `OSWORLD_ROOT`, or
`RSIAGENT_ENV_FILE` only when using a different layout.

Follow the setup for the benchmark you want to run. All commands below start
from the `RSIAgent` directory unless a `cd` is shown. VM images and benchmark
assets are downloaded separately.

<a id="osworld"></a>

### 🖥️ OSWorld

Install the pinned [OSWorld release](https://github.com/xlang-ai/OSWorld-V2/tree/v2026.08.08)
and follow its [Docker setup](https://github.com/xlang-ai/OSWorld-V2/blob/v2026.08.08/docs/PROVIDER_SETUP.md#docker).
From the RSIAgent checkout:

```bash
git clone --branch v2026.08.08 https://github.com/xlang-ai/OSWorld-V2.git ../OSWorld-V2
cd ../OSWorld-V2
uv sync --frozen
uv pip install --python .venv/bin/python -r ../RSIAgent/requirements.txt
source .venv/bin/activate
cd ../RSIAgent
```

Prepare the benchmark assets and audit files, then check the VM:

```bash
python tools/prepare_osworld_v2_release.py
python tools/build_p2_corpus.py
python tools/exam_fence.py build
python tools/smoke_osworld.py --output results/smoke/osworld
```

Keep this environment active for the OSWorld batch commands below. In a new
shell, activate it with `source ../OSWorld-V2/.venv/bin/activate`.
The audit files under `results/` stay on the host and never enter Agent prompts
or memory.

<a id="ale"></a>

### 🎓 ALE

The setup script clones the pinned upstream source and installs separate grader
and worker environments. This separation prevents the two projects' Python
packages from shadowing each other.

```bash
python3 scripts/setup_ale.py
../agents-last-exam/.venv/bin/python run_ale.py prepare --os linux
../agents-last-exam/.venv/bin/python run_ale.py prepare --os windows
```

Each preparation downloads only the selected OS image. Linux requires about
167 GiB and Windows about 157 GiB, plus download and VM working space. Use
`--cache /path/with/space` consistently for preparation, smoke tests, and runs.

Check both guest types before running the full CPU cohort:

```bash
../agents-last-exam/.venv/bin/python run_ale.py smoke \
  --os linux --output results/smoke/ale_linux
../agents-last-exam/.venv/bin/python run_ale.py smoke \
  --os windows --output results/smoke/ale_windows
```

ALE requires a successful smoke for each requested OS on the current source and
runner image. Use a new `--output` directory when repeating a smoke. See
[ALE operations](docs/ALE.md) for storage options and the upstream guide.

<a id="ascend"></a>

## Ascend kernel-learning campaigns

`run_ascend_campaign.py` launches one explicitly requested phase. Agents run
locally; compilation, host verification, timing, and profiling use the selected
local or remote runtime. A3 uses Ascend C; A5 retains its Catlass DSL entrypoints.
Starting Phase 1 does **not** automatically start Phase 2.

Install the host dependencies (`python -m pip install -r requirements.txt`) and
GNU Screen on the launcher host. Configure the user-wide `remote-access` skill
and the checked validation adapter, a programming guide, provider credentials,
and, when selected, KDB/profiling guidance. Check the configured target and run
a bounded smoke before committing to a campaign. No desktop VM is needed for
Ascend. See [target/runtime setup](docs/ascend-target-launcher.md) for required
native fields, treatment bindings, and explicit A3 device pools.

### Small Pi smoke with a local Ascend runtime

Run these commands **on the Ascend host**, after activating its Python/CANN
runtime and putting Node 22+ on `PATH`. Pi and kernel execution then run on that
host. Local mode does not need remote-access configuration or remote wrappers.

```bash
git clone https://github.com/mdrumond/RSIAgent.git RSIAgent
cd RSIAgent
# To test the local-execution PR instead of main:
git fetch origin refs/pull/170/head
git checkout --detach FETCH_HEAD
python -m pip install -r requirements.txt
npm ci --prefix integrations/pi
```

If the Ascend host cannot reach GitHub, create a Git bundle on a connected host
and upload it through your configured transport instead:

```bash
# In the PR checkout on the connected host:
git bundle create /private/pr170.bundle codex/pi-local-execution
cpl-remote upload YOUR_A3_TARGET /private/pr170.bundle /remote/private/pr170.bundle
# On the Ascend host, instead of the network clone:
git clone --branch codex/pi-local-execution /remote/private/pr170.bundle RSIAgent
cd RSIAgent
git rev-parse HEAD
```

Then install the Python/Pi dependencies as above. Verify that `git rev-parse HEAD`
matches the PR head you intend to validate.

Keep `DEEPSEEK_API_KEY` in your environment or a private environment file outside
the checkout. Create a private worker configuration, replacing the paths,
physical device, and compiler architecture with values for your active runtime:

```json
{
  "execution": "local",
  "workspace": "/absolute/private/pi-smoke/kernel-work",
  "physical_device": 0,
  "arch": "YOUR_COMPILER_ARCHITECTURE",
  "guide": "/absolute/RSIAgent/benchmarks/pi_coding/knowledge/ascendc.md",
  "env_file": "/absolute/private/provider.env",
  "model_profile": "deepseek-flash",
  "max_cases": 1,
  "max_seconds": 300
}
```

This small smoke asks real Pi to generate an add-one kernel for one frozen
64-element FP32 case. It uses the production optimizer, compiler, numerical
oracle, and measurement path, with a five-minute elapsed-time allocation:

```bash
export RSI_SMOKE_CONFIG=/absolute/private/worker-local.json
export RSI_SMOKE_ROOT=/absolute/private/pi-smoke/jobs
python - <<'PYSMOKE'
import json, os
from pathlib import Path
from optimizer.jobs import JobStore, canonical, sha
from optimizer.worker import load_config, run_worker

store = JobStore(Path(os.environ["RSI_SMOKE_ROOT"]))
body = {"files": {}, "memory_entries": []}
memory = {**body, "snapshot_id": sha(canonical(body))}
job = store.create(
    "def forward(x): return x + 1", "forward",
    {"signatures": [{"args": [{"shape": [64], "dtype": "float32"}]}]},
    memory, name="Pi local add-one smoke", max_seconds=300)
result = run_worker(store.root, job["job_id"],
                    load_config(os.environ["RSI_SMOKE_CONFIG"]))
states = [event["payload"] for event in store.events(job["job_id"])
          if event["kind"] == "optimizer-state"]
verified = bool(states and states[-1].get("best"))
print(json.dumps({"job_id": job["job_id"], "status": result["status"],
                  "host_verified": verified}))
raise SystemExit(0 if verified else 2)
PYSMOKE
```

Success requires `host_verified: true`; model prose and compilation alone do
not establish success. Inspect the retained job events, generated source,
operation receipts, and profiler evidence under the private run directories.
Use a fresh root for a new smoke; existing jobs retain their original allocation.
A run may end with `status: "budget-exhausted"` after using its time allocation
and still have `host_verified: true`; inspect these fields separately. A live
A3 run of this 64-element smoke used real Pi 1.3.0 and the configured provider,
compiled and numerically verified the generated kernel in local mode, and ended
at its 300-second allocation with `host_verified: true`.
For the larger fixed tiled matmul-add smoke and other launchers, see
[local kernel execution](docs/local-kernel-execution.md) and
[Pi setup](docs/PI_CODING_RUNTIME.md).

### Campaign configuration

All `/path/to/...` values below are placeholders. Keep settings, credentials,
and machine-specific configuration in ignored local files outside campaign
roots; do not commit them. There are separate configuration layers:

- **Launcher settings** (`--config` below): model, observed source manifest,
  and target CLI arguments. Phase 2 also needs `optimizer_worker_config` for
  fresh A3 optimizer-backed goals.
- **Native runtime settings** (`--config` inside `phase1_arguments`): checked
  profile/runtime, actual target/architecture, device, guide, environment
  identity, and optional KDB/profiling configuration.
- **Optimizer worker settings**: the compatible native A3 worker configuration;
  see [worker setup](docs/optimizer-worker.md). Its treatment must agree with
  the reviewed Phase 1 composition.

Example ignored launcher settings (not a complete native runtime file):

```json
{
  "model": "openai/gpt-5.6-sol",
  "source_manifest": "/path/to/observed-sources.json",
  "optimizer_worker_config": "/path/to/optimizer-worker.json",
  "phase1_arguments": [
    "--config", "/path/to/native-runtime.json",
    "--env-file", "/path/to/provider.env",
    "--model", "openai/gpt-5.6-sol",
    "--knowledge", "without-kdb",
    "--profiling", "new"
  ]
}
```

The source manifest records available repositories, exact revisions, and
licenses; it does not download repositories or weights. See
[source-manifest format](docs/a5-sft-dataset.md). Credentials belong in the
ignored environment file, not JSON or campaign output. Use a fresh root/control
pair for each new campaign; identity-bearing settings are pinned at start.

### Start, observe, and continue

```bash
python run_ascend_campaign.py start --target a3 --phase 1 \
  --root /path/to/phase1 --control /path/to/phase1-control \
  --config /path/to/launcher-settings.json --initial-memory /path/to/seed.bin \
  --objective 'Learn reliable kernels and measured performance' --project-target 64
python run_ascend_campaign.py status --control /path/to/phase1-control --watch
python run_ascend_campaign.py resume --control /path/to/phase1-control
python run_ascend_campaign.py continue --control /path/to/phase1-control --project-target 96
```

Compute and collection have separate retained Screen sessions and logs in the
control directory. `status --watch` polls every five minutes. Inspect errors and
retained remote handles before resuming; never run two controllers on one root.
`resume` retries unfinished work without increasing the target. `continue`
explicitly raises it; add `--reopen-terminal` to reopen a curriculum terminal.
Phase 1 siblings share pre-wave memory and consolidate at the wave barrier;
an A3 device pool enables parallel projects, not automatic idle-device discovery.

Before Phase 2, review the actual Phase 1 memory and evidence for usefulness,
diversity, and observed profiler guidance. The
[review contract](docs/ascend-target-launcher.md#reviewed-native-phase-2-handoff)
defines the required review JSON; do not fabricate acceptance or curate memory
to pass it. Start with the original Phase 1 initial memory:

```bash
python run_ascend_campaign.py start --target a3 --phase 2 \
  --root /path/to/phase2 --control /path/to/phase2-control \
  --config /path/to/launcher-settings.json --initial-memory /path/to/seed.bin \
  --phase1-root /path/to/phase1 --review /path/to/phase1-review.json \
  --objective 'Investigate a new kernel goal using reviewed lessons' --project-target 1
```

Fresh A3 Phase 2 selects one executable optimizer goal, freezes its reference
suite, then alternates target attempts and focused practice. Only host results
establish correctness. See [optimizer-backed Phase 2](docs/phase2-optimizer-campaign.md)
for budgets and successive goals; A5 and retained legacy runs keep their own
entrypoints. This A3 example does not establish A5 optimizer support.

Collection runs independently on poor links. To restart it or package evidence:

```bash
python run_ascend_campaign.py collect --control /path/to/phase1-control --watch
python run_ascend_campaign.py package --control /path/to/phase1-control \
  --output /path/to/phase1-evidence.tar.gz
```

Strict packaging requires a committed boundary and all referenced artifacts.
`--allow-pending` creates an explicitly incomplete checkpoint instead. Archives
bundle memory and retained evidence, not credentials or local runtime settings.
See [campaign lifecycle and collection](docs/ascend-campaigns.md).

### Learn from Phase 2 failures

The launcher supports importing observations, investigating them, and explicitly
retrying the original target. Nothing starts merely by importing results:

```bash
python run_ascend_campaign.py feedback-phase2 --feedback-root /path/to/feedback \
  --source-root /path/to/phase2
python run_ascend_campaign.py start --target a3 --phase 1 \
  --root /path/to/investigation --control /path/to/investigation-control \
  --config /path/to/launcher-settings.json --initial-memory /path/to/seed.bin \
  --feedback-root /path/to/feedback --failure-id FEEDBACK_ID \
  --objective 'Isolate the failure with smaller probes and working controls' --project-target 8
python run_ascend_campaign.py start --target a3 --phase 2 \
  --root /path/to/retry --control /path/to/retry-control \
  --config /path/to/launcher-settings.json --initial-memory /path/to/seed.bin \
  --phase1-root /path/to/investigation --review /path/to/investigation-review.json \
  --feedback-root /path/to/feedback --retry-feedback-id FEEDBACK_ID \
  --objective 'Retry the original target with reviewed investigation memory' --project-target 1
```

Read `FEEDBACK_ID` from the imported records or dashboard. Review the investigation
before issuing the retry command. Investigation success does not resolve the
original failure: only a successful original-task retry does. Retries require
fresh directories and newly reviewed memory, preserve the original expanded
frozen tests/comparison policy, and currently support A3 executable optimizer
tasks. Slow kernels must meet their original performance goal. Full profiler
trees stay remote; feedback contains bounded summaries and evidence references.
See [feedback observations](docs/phase2-feedback.md) and
[failure-directed curriculum](docs/failure-directed-curriculum.md).

<a id="dashboard"></a>

## Campaign dashboard

Create an ignored local JSON using [the example](dashboard/config.example.json),
then install and start the viewer:

```bash
python -m pip install -r dashboard/requirements.txt
python -m dashboard serve --config /path/to/dashboard.json --cache-dir /path/to/dashboard-cache
```

Open **http://127.0.0.1:8080**. In `campaigns`, list locally accessible campaign
`root` paths, phase/target, and launcher `control` directories. Set top-level
`feedback_root` to the shared feedback store; configure campaign settings with
`phase2_feedback_root` to publish future Phase 2 observations there rather than
the default `CAMPAIGN/feedback`. The viewer shows progress, retained traces,
source, host verification, timing/profiling evidence, and failure/retry lineage.
Missing evidence is shown as unavailable, not reconstructed.

Controls are **off by default**. For optional controls, keep the listener on
loopback, set `controls.enabled`, configure the launcher `runner`, and define
complete server-side launch `presets` with unused root/control directories.
With controls enabled, **every campaign and preset in that configuration must
explicitly select `target: "a3"`**; mixed A3/A5 configurations are not supported.
To view A5 alongside an A3 control dashboard, run a separate controls-disabled
instance with its own configuration, cache directory, and port (for example,
`--port 8081`). Phase 2 presets also require reviewed `phase1_root` and `review`
paths. Use an SSH tunnel for remote access.

Controls can sample/approve curriculum, launch approved projects, continue
eligible campaigns, import feedback, start investigations, and request retries.
Viewing/importing feedback never launches agents; every investigation/retry
needs a fresh preset. Infrastructure/provider observations are retryable but
not kernel-investigation tasks. The **Kernel Optimizer** tab is separate: its
own `optimizer` configuration enables individual submission jobs, not campaign
controls. See [dashboard setup and controls](dashboard/README.md) for configuration,
approval/recovery behavior, and optimizer setup.

<a id="run-batches"></a>

## 🚀 Run batches

Choose `--arm baseline` for task execution without RSI, `--arm rsi` for learning
followed by frozen-memory evaluation, or `--arm both` to run both.

<a id="osworld-1"></a>

### 🖥️ OSWorld

Inspect the full 108-task plan, then run it:

```bash
python run_osworld.py --arm both --name osworld_run_01 --dry-run
python run_osworld.py --arm both --name osworld_run_01 --concurrency 1
```

`--dry-run` prints the plan without starting VMs, making model calls, or writing
outputs. Logs and task status are under `results/batches/<name>/`. A task failure
stops its remaining phases; other tasks continue. Choose a new `--name` for each
batch because existing outputs are never overwritten.

<a id="ale-1"></a>

### 🎓 ALE

Inspect the cohort, run the supported tasks, and generate a report:

```bash
../agents-last-exam/.venv/bin/python run_ale.py plan --arm both
../agents-last-exam/.venv/bin/python run_ale.py run \
  --arm both --output results/ale/run_01 --concurrency 1
../agents-last-exam/.venv/bin/python run_ale.py report \
  --runs results/ale/run_01 --output results/ale/report_01
```

Use a new `--output` directory for each run. The report contains `tasks.csv` and
`summary.json`, keeps missing results explicit, and rejects
duplicate scored attempts.

Both entrypoints run batches directly. Begin with `--concurrency 1`; raise it
when the host has capacity for additional independent task lineages. For custom
OSWorld protocols and recovery, see [operations](docs/OPERATIONS.md).

<a id="validation"></a>

## ✅ Validation

Portable checks run without credentials, Docker, or benchmark installations:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python tools/check_rsi_release.py
```

The VM smoke commands in the installation steps check transport, immutable
memory, candidate replay, Verifier isolation, and checkpoint rollback using
synthetic files. They make no model or official grader calls. Smoke success
validates runtime mechanics; reproducing benchmark scores requires complete
experiments with the pinned configuration.

<a id="documentation"></a>

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Pi production coding runtime and setup](docs/PI_CODING_RUNTIME.md)
- [Historical Pi versus RSI coding evaluation](docs/PI_CODING_EVALUATION.md)
- [OSWorld operations and recovery](docs/OPERATIONS.md)
- [ALE setup, batches, and reports](docs/ALE.md)
- [Ascend campaign launcher](docs/ascend-campaigns.md) · [Target/runtime setup](docs/ascend-target-launcher.md)
- [Phase 2 optimizer workflow](docs/phase2-optimizer-campaign.md) · [Failure-directed learning](docs/failure-directed-curriculum.md)
- [Campaign dashboard and controls](dashboard/README.md)
- [Release provenance and validation](docs/RELEASE.md)
- [Paper and reporting scope](docs/PAPER.md)
- [Contributing](docs/CONTRIBUTING.md) · [Third-party attribution](docs/THIRD_PARTY.md)

<a id="citation"></a>

## 📝 Citation

If you use RSIAgent, please cite the [arXiv preprint](https://arxiv.org/abs/2609.15364):

```bibtex
@misc{zhu2026rsiagentautonomousexplorationrecursive,
      title={RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments},
      author={Sibo Zhu and Shicheng Fan and Xinyue Wang and Wenyi Wu and Kun Zhou and Biwei Huang},
      year={2026},
      eprint={2609.15364},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.15364},
}
```

<a id="license"></a>

## ⚖️ License

RSIAgent is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies and benchmark assets retain their respective licenses and terms; see [third-party attribution](docs/THIRD_PARTY.md).
