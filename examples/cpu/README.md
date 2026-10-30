# CPU foundation fixture

From the repository root:

```bash
python -m rsi validate-spec examples/cpu/cycle.json
(cd examples/cpu && python verify.py)
```

This is a known-good numeric task, not a workflow runner. Replacing `math.fsum`
with naive sequential addition creates a cancellation failure detected by the decimal verifier.
The baseline references are placeholders; this fixture does not open a KDB or
call a provider. Later workflow PRs supply execution and evidence enforcement.

Install the minimal CLI with `pip install .`. Existing application packages are
included in the distribution; `pip install '.[legacy]'` also installs their
existing pinned host requirements. CPU examples remain repository examples.
