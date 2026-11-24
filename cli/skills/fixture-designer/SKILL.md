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

Choose the next action from the evidence in the packet.

- **Test** when you can identify an executable failure, a bounded observable and independent checks. For a buffer transformation, vary the values that should change and retain sentinels around the valid region. An exit code alone leaves the transformation unchecked.
- **Probe** when a concrete observation could change your design. For example, inspect the installed signature if two plausible call forms differ by version. Use the result to settle that question. An identical rerun after a syntax error would leave the original question open.
- **Withhold** when an indispensable source or observable is missing and the available tools cannot recover it. Name the missing evidence and its basis. A missing private reference answer can often be addressed by deriving a small oracle from the public contract.

Select the diagnostic by its supplied reference. Use a stable numerical check identifier and report the observed and expected values on failure. In a revision, preserve useful controls and address the specific measured discrepancy.
