# Pi versus RSI coding microbenchmark

This opt-in microbenchmark compares the RSI coding-loop slice with a persistent Pi coding session on a fixed Ascend C matmul-plus-add task. Both receive the same task, starter, model settings, starting knowledge, and independent A2 validator. RSI retains ownership of verification and spending. This does not replace RSI's curriculum, memory, or production actor.

## Observed result

The reference-assisted DeepSeek trial on 2026-10-16 produced the following result:

| Backend | Result | Input tokens | Output tokens | Total tokens | Requests | Revisions | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RSI coding loop | Compile failure at limit | 49,412 | 24,273 | 73,685 | 13 | 12 | 20m 11s |
| Persistent Pi | All numerical cases and source review passed | 20,432 | 14,080 | 34,512 | 5 | 4 | 9m 58s |

Pi succeeded using 53.2% fewer generation tokens than RSI spent in this pair. RSI's tokens-to-success are unknown because it did not succeed. Cached input tokens are already included: 35,968 for RSI and 9,600 for Pi. Wall time includes generation, transport, compilation, and validation; it is not kernel execution time.

Pi repaired its initial missing kernel-type declaration with a targeted edit, then changed its transfers after the irregular shapes failed. The final kernel uses bounded local buffers, 64×64 output tiles and 32-wide reduction chunks. All tested elements matched the reference exactly. It uses scalar arithmetic on the Ascend C AIV path, not an optimized cube matmul. No throughput claim is made.

This is one paired trial, executed RSI first and Pi second, without manual candidate fixes. It supports further evaluation of Pi for complex implementation. It does not establish average superiority or independently prove the benefit of starting knowledge. The earlier Pi attempt without references was interrupted, so it cannot serve as a completed ablation.

The [sanitized evidence bundle](../benchmarks/pi_coding/evidence/2026-10-16/README.md) includes per-request usage, validation outcomes, hashes, and the passing kernel. Its kernel is a review/replay artifact and is never loaded into generation prompts. Raw sessions, prompts, credentials, diagnostics, remote jobs and connection configuration are excluded.

## Setup

Use Python 3.12, Node.js 22 or newer, and the pinned Pi runtime. Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm ci --prefix integrations/pi
mkdir -p .codex/local/pi-investigation
cp benchmarks/pi_coding/examples/providers.env.example .codex/local/pi-investigation/providers.env
chmod 600 .codex/local/pi-investigation/providers.env
cp benchmarks/pi_coding/examples/connection.example.json .codex/local/pi-investigation/connection.json
```

Fill the ignored provider file with `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` using your editor. Do not put credentials in commands, prompts or tracked files. The provider file must have mode 0600. The host gateway passes only temporary local tokens to Pi; provider keys are absent from Pi's environment and from remote scripts.

Live validation requires an already configured laptop-managed remote-agent API and an accessible A2 target. Its access helper is an external prerequisite, not bundled infrastructure. Follow your remote-access playbook to verify health, target `a2`, session configuration, listener/asyncssh transfer support, CANN, and an idle mapped device. Fill the template's helper path, CANN environment script, NPU Python executable, compiler architecture and device index. An optional `container` field names an existing, preflighted container. The tested environment used CANN 9.1 and PyTorch/torch_npu 2.10 on an Ascend 910B4-1. Other installations require their own preflight.

All remote operations go through the helper, which resolves the current API endpoint. There is no direct SSH fallback, UID override, port scan or service/container lifecycle operation. If transport fails, inspect retained job IDs and repair laptop connectivity before retrying. A timed-out submission may still be running; never blindly resubmit it. Machine-specific configurations and generated execution scripts stay under ignored `.codex/local/` state.

## Run a paired comparison

```bash
.venv/bin/python -m benchmarks.pi_coding.compare \
  --budget-usd 10 \
  --connection .codex/local/pi-investigation/connection.json
```

New runs require an explicit total budget in `(0, 10]` USD, divided equally between arms. The default provider is `deepseek` (`deepseek-flash`, thinking disabled); `--provider openai` selects `gpt-5.6-sol` with high reasoning using Responses for both arms. Provider access is not guaranteed by the benchmark. The runner checks the board fixture and makes one small billed provider preflight before generation. Its usage is separate from generation totals and charged to the RSI allocation.

Each arm starts from a fresh TODO kernel and stops on a numerical pass, infrastructure/provider failure, its budget, 12 candidate revisions/validations, 30 generation requests, or 60 minutes. Requests allow up to 16,384 output tokens, including reasoning where applicable. Both backends use compact host feedback. A successful run still needs source review for tiling and the task restrictions. Exit code 0 means both arms passed numerical validation; 2 means at least one failed, stopped at a limit, or could not run. Inspect the report for the distinction.

The task is `D[M,N] = A[M,K] @ B[K,N] + C[M,N]`, row-major FP16 A/B, FP32 C/output/accumulation. One block handles a 64×64 output tile; K is processed in chunks of 32. The fixed shapes are `(64,64,64)`, `(129,193,257)` and `(257,321,193)`: 1/12/30 output tiles and 2/9/7 reduction chunks. The trusted validator checks every output element with `rtol=atol=1e-4`, rejects nonfinite or missing outputs, and binds results to the source SHA-256. No CPU/host matmul fallback or precomputed answer is permitted. Candidate tools cannot edit the trusted driver or binding; this is not an OS security sandbox for hostile generated code.

### Starting knowledge and custom requests

Both arms use [portable Ascend C notes](../benchmarks/pi_coding/knowledge/ascendc.md) by default. Replace them with `--knowledge-file FILE`, or omit them with `--no-knowledge`. An optional quoted request or `--prompt-file FILE` adds implementation instructions within the fixed task contract:

```bash
.venv/bin/python -m benchmarks.pi_coding.compare \
  --budget-usd 10 \
  --connection .codex/local/pi-investigation/connection.json \
  --knowledge-file benchmarks/pi_coding/knowledge/ascendc.md \
  'Handle every partial M/N tile and K tail using bounded working buffers.'
```

The combined request and knowledge must fit within 16 KiB. Both arms receive identical composed inputs. The report records hashes for the task, starter, request and knowledge; each run also records its complete prompt hash. Knowledge is explicitly supplied, not discovered from arbitrary files. The portable notes replace private source paths and original third-party excerpts, so new runs reproduce the methodology rather than historical prompt bytes or exact token counts. Provider nondeterminism can also change outcomes. This packaging work made no new paid model calls.

### Budgets, retained state, and resume

Runs, sessions, remote scripts and ledgers live under `.codex/local/pi-investigation/`. Each fresh comparison gets its own allocation. To rerun both arms using only an existing allocation's remaining budget:

```bash
.venv/bin/python -m benchmarks.pi_coding.compare \
  --reuse-budget-from COMPARISON_NAME \
  --connection .codex/local/pi-investigation/connection.json
```

The previous comparison must be terminal. The new report links to the original ledger, including when reusing a previously reused allocation. Concurrent paired runs sharing the allocation are rejected. Prior requests and unknown-usage reservations are never reset. Budget reuse does not resume candidate code: both arms restart independently.

For a comparison blocked before generation, `--resume COMPARISON_NAME` preserves its saved prompt and knowledge. It requires a previously passing board preflight and no started generation arm. It checks live transport and reruns the fixture when the connection changed. A provider switch is explicit through `--provider`; do not use resume to restart an ambiguous generation job.

Cost accounting uses conservative fixed rates of $2/$4 per million input/output tokens for DeepSeek and $8/$40 for OpenAI, not provider invoices or dynamically fetched prices. Unknown usage retains its full request reservation. Reports distinguish phase spending from cumulative allocation bounds. The published phase bounds were $0.195938 for RSI, including preflight, and $0.097184 for Pi. Earlier attempts and a retained rejected OpenAI reservation brought that allocation's cumulative bound to $0.463276 of $10. The earlier $3 smoke allocation was separate and was not reset.

## Inspect or replay the published kernel

This makes no provider calls; it requires the same A2 prerequisites:

```bash
.venv/bin/python -m benchmarks.pi_coding.validate_saved \
  --source benchmarks/pi_coding/evidence/2026-10-16/pi-kernel.cpp \
  --expected-sha256 ab607cf180a69b388f51fe68b53e226d4cc9e7891b6acd1c4b032ea7b8022f61 \
  --connection .codex/local/pi-investigation/connection.json
```

Replay writes a new local validation record without changing the evidence bundle. `--run LOCAL_RUN_NAME` remains available for private saved runs and verifies the recorded source hash. The trusted validator remains fixed; a custom user request does not redefine it.

To export a completed local comparison for review:

```bash
.venv/bin/python -m benchmarks.pi_coding.export_evidence \
  --comparison COMPARISON_NAME \
  --output .codex/local/pi-investigation/review-export \
  --include-passing-source
```

The exporter constructs a new typed, allowlisted metrics document rather than copying raw records. Passing-source export is opt-in, checks the verified source hash and common private-data patterns, and still requires human source review before publication. It refuses an existing output directory. Do not publish raw comparison directories.

## Implementation and offline checks

The paired CLI owns preflights and allocation locking; the single-run host owns deadlines, candidate tools and validation. `backends.py` implements RSI's existing `complete_a3`/`parse_action` route and Pi's JSONL RPC lifecycle. Accounting, remote transport, numerical verification and evidence export are separate modules. The Pi npm dependency is optional and pinned to 1.3.0 with a lockfile; no production RSI entrypoint selects it automatically.

Persistent Pi keeps one process/session and exposes `read_candidate`, `submit_candidate`, and exact-text `edit_and_validate`. Its context hook retains the task, current source, latest diagnostic, brief earlier failures and the latest complete tool exchange. Raw history remains local. Automatic paid compaction, cache warming, transport retries, shell tools and arbitrary file access are disabled. The RSI baseline regenerates complete source through its JSON action protocol. Those differences are part of the coding-loop comparison, not identical orchestration.

`python -m benchmarks.pi_coding.experiment` remains the advanced single-run entrypoint. Its older `native` and `efficient` Pi modes are legacy smoke workflows; the paired microbenchmark always uses `persistent`. Its default single-run limits remain three revisions and ten minutes for compatibility.

```bash
.venv/bin/python -m pytest -q \
  tests/test_pi_coding.py tests/test_pi_coding_tasks.py \
  tests/test_pi_coding_efficiency.py tests/test_pi_coding_persistent.py \
  tests/test_pi_coding_microbenchmark.py
.venv/bin/python tools/check_rsi_release.py
```

Offline tests use fake providers and validators, with real installed Pi processes for RPC/extension behavior. They need no provider credentials or board connection. Real-Pi tests skip when the optional package is absent; the dedicated CI job installs it and executes them. Relevant regressions cover existing A3 model routes and trial behavior. Tests cover spend limits, usage subsets, unknown reservations, immutable task inputs, targeted edits, truncation, numerical tails, source hashes, remote ambiguity, allocation locking and evidence privacy.

## Historical context and next questions

The original vector-add repair was a wiring smoke: both backends repaired the same trivial error, and native Pi consumed more calls and tokens. A later three-revision matmul trial produced no correct kernel. Before references were supplied in the full trial, RSI spent 37,400 generation tokens without passing; Pi was intentionally interrupted at 9,131 tokens. These development attempts are distinct from the completed reference-assisted pair and are not averaged into it.

Future evaluation should repeat matched tasks and compare both backends with and without starting knowledge. Production adoption also needs explicit integration with RSI's verified memory, curriculum and verifier ownership. This microbenchmark does not measure those responsibilities or establish that Pi replaces the complete RSI framework.

Packaging verification on 2026-10-16: the repository-wide release check reported 2,941 passed, 4 skipped, and one pre-existing failure in `test_no_kdb_catlass_opening_specifies_executable_contract`. That A5 test expects the phrase “all P output elements” while the implementation emits “all 64 output elements”; the same failure was reproduced in a clean checkout of the baseline revision. The Pi microbenchmark tests passed. No new provider calls or board runs were made during packaging.
