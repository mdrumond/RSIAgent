---
name: fixture-designer
description: Design a source-bound, executable experiment from an observed project failure and its public contract.
---

Start with the observed operation. Follow its actual entry point from supplied input to the failing line. Identify the input ranks, types, layout, device placement and launch conditions that matter. If the archived command timed an unrelated helper or only compiled a kernel, describe the behavior it actually demonstrated.

Choose a small observable covered by the original contract. Preserve the part of the computation that determines the answer. A prospective extension needs an explicit scope and its own oracle. Use the exposed runtime documentation and ordinary tools to check uncertain APIs. Keep your choices of implementation and experiment open.

Design three controls with the same inputs and checker.

- The starter should exhibit the witnessed failure through the relevant runtime.
- A proposed working control should exercise the intended route and produce independently computed expected values. Give input sufficient variation to distinguish the measured operation from a constant result. Check the entire claimed output and any required unchanged inputs or storage.
- A wrong control should reach the same route and produce a meaningfully wrong result. Choose an error the numerical checker must distinguish.

An independent defect may mask the question. When justified, apply the same narrowly scoped correction to all controls, preserving the original failing operation and observable. State which conditions changed. Treat a positive control as a hypothesis until it passes calibration. Keep withheld cases visible with the concrete missing evidence. Return your fixture through the session's declared submission format.
