# Desktop benchmark setup

For OSWorld and ALE, use Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a Linux host with Docker and access to `/dev/kvm`. Start with the repository
and your own model API credential:

```bash
git clone https://github.com/mdrumond/RSIAgent.git
cd RSIAgent
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY` in `.env`. Paths default to this checkout and sibling
benchmark directories. Export `RSIAGENT_ROOT`, `OSWORLD_ROOT`, or
`RSIAGENT_ENV_FILE` only when using a different layout.

Follow the setup for the benchmark you want to run. All commands below start
from the `RSIAgent` directory unless a `cd` is shown. VM images and benchmark
assets are downloaded separately.

## OSWorld

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
python tools/build_practice_corpus.py
python tools/exam_fence.py build
python tools/smoke_osworld.py --output results/smoke/osworld
```

Keep this environment active for the OSWorld batch commands below. In a new
shell, activate it with `source ../OSWorld-V2/.venv/bin/activate`.
The audit files under `results/` stay on the host and never enter Agent prompts
or memory.

## ALE

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
[ALE operations](ALE.md) for storage options and the upstream guide.


For batch commands and reporting, return to the [README](../README.md#desktop-benchmarks), [OSWorld operations](OPERATIONS.md) or [ALE guide](ALE.md).
