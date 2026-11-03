# Ascend controller

A native Pi controller for Ascend kernel research. Node owns project admission,
attempt state, elapsed budgets and the canonical workflow. Python provides the
knowledge database, MCP service, trusted target verification and a thin local
Flask dashboard. Pi owns model requests, tools, sessions and context management.

This source supports fresh Ascend instances only. The former RSI engines,
non-Ascend applications, release-adoption tools and historical-state migrations
are removed. Existing research evidence and baselines remain separate inputs;
old campaign state is never loaded into a new instance.

## Install

Use Node **22.23.3**. Python **3.12** is the tested, recommended host; the helper
package permits Python **3.11+**. Full host runtime acceptance on Python 3.11 is
not yet established. Target kernels use their registered Python **3.11** runtime
and pinned dependencies independently of the host installation.

```bash
npm ci
python -m pip install -e .
# Only for production BGE embedding queries:
python -m pip install -e '.[embeddings]'
```

The standard npm shrinkwrap pins Pi 1.3.0. The Python package installs only `ascend_tools`,
with MCP 1.27.2 and Flask 3.0.3. The optional embedding runtime pins Torch 2.10.0
and Transformers 5.0.0. The Python wheel contains helpers; the Node CLI and web
assets are supplied by the source machinery and copied into each instance.

## Create an instance

Prepare a private bootstrap JSON file containing `baseline` (an existing KDB
SQLite file), `skills` (a baseline skills directory), and `environment` (extension
module, access instructions and opaque execution options). The destination must
be a new directory outside the machinery checkout.

```bash
node cli/main.mjs bootstrap "$INSTANCE" --config "$BOOTSTRAP_JSON"
node "$INSTANCE/machinery/cli/main.mjs" config --instance "$INSTANCE"
node "$INSTANCE/machinery/cli/main.mjs" project --instance "$INSTANCE" --project "$PROJECT_JSON"
node "$INSTANCE/machinery/cli/main.mjs" status --instance "$INSTANCE"
```

All variables above refer to user-selected paths. Keep credentials, machine
paths and target runtime settings in private configuration outside this source.
Bootstrap automatically installs copied-machinery dependencies with `npm ci`,
using the copied standard `npm-shrinkwrap.json`. If that installation needs
repair, rerun `npm ci --ignore-scripts --prefix "$INSTANCE/machinery"`.

Actor profiles use root `actor`, `curriculum_actor` and `skill_writer_actor`
fields in bootstrap/settings JSON. Use `config --config "$SETTINGS_JSON"` to
write settings, then start a bounded wave when ready:

```bash
node "$INSTANCE/machinery/cli/main.mjs" wave --instance "$INSTANCE" --mode phase1 --detach
node "$INSTANCE/machinery/cli/main.mjs" stop --instance "$INSTANCE"
node "$INSTANCE/machinery/cli/main.mjs" resume --instance "$INSTANCE" --detach
node "$INSTANCE/machinery/cli/main.mjs" dashboard --instance "$INSTANCE"
```

The dashboard binds to `127.0.0.1:46731`. It edits settings and submits structured
CLI requests; it owns no scheduler or execution engine. Full research campaigns
remain a separate, explicitly requested operation.

See [the controller guide](docs/ascend-controller.md) for configuration, workflow,
recovery, evidence and CPU validation. Target-specific verifier contracts are
in [the target guide](examples/ascend-targets/README.md).

## CPU validation

After integrating the new controller, CLI and helper streams:

```bash
python -m pip install -e '.[dev]'
PYTHON=python PYTHONPATH=.:tests/ascend node --test --test-concurrency=1 tests/ascend/*.mjs
python -m pytest -q tests/ascend
```

Run these commands serially. Portable tests use deterministic providers,
embeddings and mocked devices; they do not establish A3/A5 device correctness or
timing. Real model, baseline and device validation must retain its own evidence.
