# Direct-agent context budgets

The shared direct provider rejects requests exceeding a local hard byte budget
before transport. This covers Actor, curriculum, knowledge and distillation
callers. Limits measure UTF-8 JSON request bytes (including message framing), not
provider token windows. Canonical model IDs and output settings are unchanged.

| Model | Compact above | Target after folding | Hard limit |
| --- | ---: | ---: | ---: |
| OpenAI | 128 KiB | 80 KiB | 192 KiB |
| DeepSeek | 32 KiB | 24 KiB | 48 KiB |

DirectActorDriver keeps the immutable opening, current host-origin source/status
facts and recent exchanges separate from a model-written working log. Earlier
exchanges are summarized in bounded batches. A rewritten candidate clears previous
candidate status from pinned host state. Logs cannot establish correctness; the
host verifier and publication rules remain unchanged. A large immutable opening,
unfoldable exchange, failed summary or oversized work log raises the explicit
non-infrastructure `ContextBudgetError`, rather than retrying device work.

Configure limits when constructing the shared provider, for example:

```python
from benchmarks.ascend.agent_context import ContextBudget
from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider

provider = DirectCompletionProvider(environment, context_budgets={
    "deepseek-flash": ContextBudget(high=40 * 1024, target=28 * 1024, hard=56 * 1024),
})
```

Summary requests/responses are retained with `stage=compaction`; their completion
tokens count in Actor usage, but not action turns or kernel trial budgets. Full
chronological messages and original evidence remain stored. Trace tool events
`provider-request` contain the exact wire request, and `context-compaction` records
the model log transition. A chronological training conversation is therefore not
necessarily identical to the bounded context supplied on a particular turn; use
these request events or the request CAS artifacts to reconstruct actual inputs.

Terminal distillation also applies the budget and bounded host evidence views.
Complete evidence stays in content-addressed storage, not in repeated prompts.
Its schema and admitted citation map remain pinned through correction attempts,
outside the work log; only the latest invalid response/correction is retained in
the correction tail. A provider outage during distillation retains usage from
successful summaries. Context-budget aborts are recorded as failed local Actor
validation with `abort_status=context-budget-exhausted`, not infrastructure loss.

Run focused regressions with:

```bash
python -m pytest -q tests/test_agent_context.py tests/test_a5_phase1_provider.py
```
