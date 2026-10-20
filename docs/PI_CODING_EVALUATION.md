# Pi coding evaluation

The portable matmul-plus-add task, knowledge notes, accounting helpers, and historical evidence live in `benchmarks/pi_coding`. The historical run used a prototype execution adapter which is intentionally excluded from this branch. New production validation must use the existing Ascend execution interfaces.

The retained October 16 comparison measured 34,512 tokens for Pi and 73,685 for the original RSI loop. Pi passed all three numerical cases; RSI exhausted twelve revisions with compilation failures. This is one matched task, not a general performance claim or an isolated measurement of the benefit of starting knowledge. See the [reviewable evidence](../benchmarks/pi_coding/evidence/2026-10-16/README.md).

Production setup, the capped smoke command, and migration limitations are documented in [PI_CODING_RUNTIME.md](PI_CODING_RUNTIME.md). The current production smoke is pending an existing A3 worker configuration; no fresh board result is claimed.
