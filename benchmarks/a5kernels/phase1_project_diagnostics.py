"""Adaptive Actor seam for bounded, exact-source BZ diagnostic observations."""

import json

from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionResult
from benchmarks.a5kernels.phase1_memory import HostFact
from benchmarks.a5kernels.phase1_project_executor import ProjectExecutor


def diagnostic_action(backend):
    def diagnose(payload, context):
        kind = payload.get("kind")
        if kind == "request-debug" and set(payload) == {"kind"}:
            tool = "msdebug"
        elif kind == "request-sanitizer" and set(payload) == {"kind", "tool"}:
            tool = payload["tool"]
            if tool not in ("memcheck", "racecheck", "initcheck", "synccheck"):
                return AdaptiveActionResult({"error": "unsupported sanitizer tool"})
        else:
            return AdaptiveActionResult({"error": "unsupported diagnostic action fields"})
        if context.source is None:
            return AdaptiveActionResult({"error": "write candidate source first"})
        request = ProjectExecutor.request(context.prepared, context.source)
        result = backend.diagnostic(request, tool=tool)
        digest = context.artifacts.put_json(result["artifact"])
        if digest != result["evidence_sha256"]:
            raise ValueError("diagnostic artifact hash does not match payload")
        return AdaptiveActionResult(
            {"event": "diagnostic", **result["record"], "evidence_sha256": digest},
            admitted_artifacts={digest: "diagnostic"},
            host_facts=(HostFact("diagnostic", "tool-observation:" + json.dumps(result["record"], sort_keys=True), digest),),
        )

    return diagnose
