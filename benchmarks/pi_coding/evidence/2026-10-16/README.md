# Reference-assisted DeepSeek trial — 2026-10-16

This sanitized snapshot records the completed A2 matmul-add comparison. `evidence.json` contains typed per-request usage, per-revision validation summaries, source hashes, settings and conservative cost bounds. `pi-kernel.cpp` is the exact final passing model-generated source, SHA-256 `ab607cf180a69b388f51fe68b53e226d4cc9e7891b6acd1c4b032ea7b8022f61`. It is an evaluation artifact, never a generation input.

| Backend | Outcome | Generation tokens | Revisions |
| --- | --- | ---: | ---: |
| RSI coding-loop slice | Compilation failed at limit | 73,685 | 12 |
| Persistent Pi | All three numerical cases passed | 34,512 | 4 |

Manual source review confirmed 64×64 output tiles, 32-wide K chunks, bounded A/B/accumulator/output buffers totaling 24,832 bytes, FP32 accumulation and addend/output, edge handling, and no host fallback. Arithmetic is scalar on the AIV path. All tested elements matched exactly; throughput and additional input distributions were not evaluated.

The task started from a TODO kernel, not this solution. Both arms used the same model/settings, prompt, references, validator and limits. RSI ran first, then Pi. Pi's second and third revisions compiled but failed irregular-shape validation; its fourth revision staged A/B tiles and used padded output transfers. RSI remained stuck on a DataCopyPad overload error. No generated candidate received a manual fix.

This historical prompt included third-party excerpts and private source locations. It is intentionally excluded. The public starting-knowledge file is an independently written portable summary, not a byte-identical replacement. The original composed request hash and per-arm complete prompt hashes are retained. Separate knowledge hashes and implementation fingerprints were not recorded by that prototype and remain null rather than being reconstructed as historical facts. The published numerical contract is unchanged. Limits were 12 revisions, 30 generation requests, 60 minutes per arm, and 16,384 output tokens per request; the latter is recorded in each arm's generation settings.

The prior allocation bounds were $0.138656 for RSI and $0.031498 for Pi. This phase added $0.195938 and $0.097184 respectively, yielding $0.463276 total. These are conservative bounds including previous attempts, provider preflights and retained unknown reservations, not invoices. Generation tokens exclude preflights and previous attempts. Cached tokens are subsets of input totals. The preflight usage field describes this phase only.

One pair supports a follow-up evaluation, not an average quality/cost claim or a causal claim about knowledge. The RSI baseline did not succeed, so its tokens-to-success are null. See the [benchmark guide](../../../../docs/PI_CODING_EVALUATION.md) for historical limitations and the separate production smoke workflow.
