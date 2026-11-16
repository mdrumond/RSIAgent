# Model context allowance

Actor profiles accept `max_context_tokens`, an integer of at least 2. It reduces
native Pi's model `contextWindow` to the smaller of this value and the provider's
advertised capacity. For a 200k experiment, set `max_context_tokens: 200000` on
every participating actor profile (including `skill_writer_actor`). Existing
profiles without the field retain provider defaults.

Native model restoration and compaction use the reduced capacity. This delegates
context management to Pi's token estimates and compaction; it does not add an
exact-token guard around provider requests. An oversized first prompt or tool
result still needs bounded inputs and provider overflow handling. Set an explicit
`max_tokens` output allowance as well; the model's output capacity is clipped
below the effective context window.

Microevaluation manifests freeze the configured context allowance. A resumed
comparison or continuation cannot change it. Start a separate experiment to
compare context allowances; preserve the original experiment's elapsed ledger.
