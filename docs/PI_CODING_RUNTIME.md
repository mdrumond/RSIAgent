# Pi for Ascend coding

Pi 1.3.0 is the production coding engine for the older A3 trial loop, shared Ascend/A5 campaign actors, Phase 2 source generation, and the optimizer. RSI still owns curriculum, knowledge permissions, memory, native execution, numerical verification, and profiling. Desktop and OSWorld agents are unchanged. There is no automatic fallback to the former direct coding loop.

## Installation and use

Use the repository virtual environment and Node 22 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm ci --prefix integrations/pi
```

Existing campaign and optimizer commands keep their host execution interfaces. Supply provider credentials through the existing ignored environment file or environment loader. Keep credentials, machine configuration, sessions, generated scripts, and raw traces outside tracked files. Pi runtime dependencies are ignored. Missing or mismatched Pi/Node installations fail locally.

New optimizer configurations default to `deepseek-flash`, with thinking disabled and a maximum of 8,192 output tokens per request. Explicit registered model selections remain authoritative. OpenAI uses the registered GPT profile through the Responses API; offline tests cover this path, but this change has no new live OpenAI validation. The older A5 pilot retains its fixed GPT model label and now uses the registered direct OpenAI route through Pi, requiring that route's local credential.

Run an existing optimizer job with its existing private worker configuration:

```bash
.venv/bin/python -m optimizer.worker \
  --root /absolute/ignored/jobs \
  --job-id JOB_ID \
  --config /absolute/ignored/worker.json
```

Do not resume an in-progress direct-coding attempt as a Pi attempt. Start a new attempt/campaign when its engine identity differs. Completed evidence remains available to existing report readers. Pi version, adapter/context policy, and model identity are recorded; optimizer identity also binds its guide and provider-request limit. The host checkpoint remains authoritative on resume. Retained optimizer actions and uncertain execution handles retain their existing reconciliation rules.

## How actions work

The host supplies a task snapshot and a set of allowed typed tools. Pi can page admitted context records, select the next host action, and make exact-text edits against a source hash. A tool receipt only queues an action. The host then validates and executes that action and supplies the resulting observation on the next turn. Model prose cannot establish compilation, correctness, or timing.

Each logical attempt owns its Pi process. The optimizer uses a persistent supervised child across host actions; Stop, deadlines, worker disappearance, and normal loop exit terminate its process group. Project sessions remain separate under parallel campaign execution.

The extension disables built-in filesystem/shell tools, automatic retries, automatic paid compaction, cache warming, external context discovery, skills, and MCP discovery. Pi receives an isolated home directory and temporary loopback credentials, while model credentials remain in the Python gateway. This is a restricted tool interface, not a general operating-system sandbox.

The host validates both the executed arguments and the original assistant arguments because Pi can coerce scalar types. Wrong-stage tools, ambiguous/stale edits, truncated output, and multiple mutations do not release a queued host action. Existing parsers and execution gates remain the final authority.

## Context and accounting

Large guides, source, memory, and observations become hash-addressed admitted records with previews. `read_context` returns at most 8 KiB with an explicit continuation offset. The latest complete tool exchange is retained within a host turn; superseded turns are replaced by the authoritative host snapshot. This avoids replaying entire source revisions and diagnostics.

DeepSeek snapshots target 16 KiB, leaving space inside the existing 24 KiB target / 32 KiB high / 48 KiB hard production context envelope for the system prompt, tool schemas, framing, and one page exchange. The gateway measures the complete serialized outgoing request and refuses anything above 48 KiB before upstream HTTP, independently of Pi's context hooks. OpenAI retains the larger registered hard envelope. Mandatory snapshots that do not fit fail locally; no model call is used to summarize them.

Pre-HTTP request intentions and provider results are retained separately. Page reads, invalid calls, and truncated generations consume the provider-request allowance. Unknown usage remains unknown. Cached input and reasoning tokens are subsets of provider totals and are not added twice. Optional dollar budgets reserve conservatively before HTTP and retain unknown reservations. Rates in the smoke ledger are conservative accounting assumptions, not invoices.

Noncoding terminal distillation still uses the existing provider, admitted-evidence parser, and bounded correction loop. It receives terminal host state and admitted artifacts; oversized original instructions are represented by a digest and bounded preview.

## Capped matmul-add smoke

The new smoke calls the existing production A3 optimizer. It introduces no remote execution adapter. It requires an already configured worker, guide, provider environment, and original allocation ledger. The three fixed `(M,N,K)` cases are `(64,64,64)`, `(129,193,257)`, and `(257,321,193)`. The task requests 64×64 output tiles, K chunks of 32, FP16 A/B, FP32 accumulation/C/output, and edge/reduction-tail handling. It uses the production `optimizer_kernel` ABI and frozen verification policy; the historical prototype used a different wrapper.

```bash
.venv/bin/python -m benchmarks.pi_coding.smoke \
  --worker-config /absolute/ignored/worker.json \
  --budget-ledger /absolute/ignored/original-pi-budget.json \
  --root .codex/local/pi-production-smoke \
  --window pi-production-smoke-v1 \
  --budget-usd 2 \
  --knowledge-file benchmarks/pi_coding/knowledge/ascendc.md \
  --check-only
```

Remove `--check-only` to execute after checking the existing worker configuration. Local preflight does not attest remote readiness. Reuse the same root, window, and ledger for a resumed run. The original ledger limit is preserved; the incremental window cannot be silently increased. Limits are 20 provider requests, four evaluated candidates, 20 minutes, and at most $2 additional conservative spend. Inspect the generated source for tiling even after a numerical pass. Raw job records and the smoke summary remain local.

The new production live smoke was **not run**: no existing A3 optimizer worker configuration was available, and the owner confirmed it is unavailable for now. No new model credits were spent. Historical numerical results are not presented as validation of this production migration.

## Review and validation

Runtime code lives in `coding/pi/`; the sole extension is `integrations/pi/host-actions.ts`. Production adapters never import runtime code from the microbenchmark package. The prototype's remote adapter and connection example were removed from the introduced branch history. Existing main-branch execution adapters remain in use.

The historical evidence and portable knowledge are described in [PI_CODING_EVALUATION.md](PI_CODING_EVALUATION.md). The one historical pair supports investigating Pi, not a general superiority or cost claim.

```bash
.venv/bin/python -m pytest -q tests/test_pi_runtime.py tests/test_optimizer_provider.py
.venv/bin/python tools/check_rsi_release.py
.venv/bin/python -m pytest -q dashboard/tests
```

Real Pi tests use fake DeepSeek/OpenAI providers and exercise persistent sessions, strict typed actions, exact edits, context paging, full-request byte limits, truncation, accounting, parallel isolation, and RPC failure cleanup. Supervisor tests exercise persistent state and cancellation of descendants. Existing host-loop regression tests continue to cover frozen cases, treatment gates, verification, distillation, and uncertain-operation recovery. Composition tests inject scripted typed sessions; they do not call paid providers.
