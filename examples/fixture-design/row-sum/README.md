# Row-sum fixture handoffs

Run `node emit.mjs OUTPUT_DIRECTORY` from this directory. The public packet contains the original task, bounded operation, failing source and diagnostic. The evaluator directory contains the proposed controls and checker. Provide each role only its worked example and the public evidence for its assigned case.

`starter.py` reproduces a scalar-load diagnostic for a rank-two input. `good.py` visits every column. `bad.py` skips the last column. The checker varies the inputs, compares against a CPU reduction and checks that the surrounding storage and input survive. Run each candidate through the replay evaluator to collect the native-route observation. `probe.py` queries the bridge signature when that specific interface is uncertain. The withhold example illustrates a different archive in which only a wrapper survived.

`packets.mjs` also exports `rowSumHandoffs()` for the three consumers. Each example supplies a proposed action, a rejected action and the feedback that explains the difference. All examples use this synthetic row sum. Historical replay cases stay in separate directories.

Run `node validate.mjs PATH_TO_FIXTURE_REPLAY_MODULE` to replay the stored measurements through the current consumer on a CPU. A fresh device run requires the Catlass runtime and the replay evaluator's instrumented launcher.
