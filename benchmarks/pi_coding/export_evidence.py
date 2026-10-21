"""Export reviewable metrics using typed allowlists, never raw run records.

Source code is an explicit opt-in artifact: inspect it before publishing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .config import LOCAL
from .report import usage_totals
from .tasks import task_contract, contract_digest

HASH = re.compile(r"[a-f0-9]{64}\Z")
STATUSES = {"passed", "incorrect", "compile_failed", "runtime_failed", "infrastructure_unavailable",
            "error", "limit_reached", "unverified"}
STOPS = STATUSES | {"revision_limit", "request_limit", "deadline", "budget_exhausted", "agent_stopped"}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def digest(value):
    return value if isinstance(value, str) and HASH.fullmatch(value) else None


def choice(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def counts(values):
    return {key: number(values.get(key)) for key in usage_totals([])}


def request_usage(row):
    """Retain only numeric usage fields; provider bodies and IDs never cross."""
    usage = row.get("usage") or {}
    clean = {key: number(usage.get(key)) for key in ("prompt_tokens", "completion_tokens")}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if number(cached) is not None:
        clean["prompt_tokens_details"] = {"cached_tokens": cached}
    if number(reasoning) is not None:
        clean["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return {"usage": clean}


def validation(row):
    clean = {"revision": number(row.get("revision")), "source_sha256": digest(row.get("source_sha256")),
             "status": choice(row.get("status"), STATUSES), "passed": row.get("passed") is True,
             "cumulative_usage": counts(row.get("cumulative_usage") or {})}
    clean["cases"] = []
    for case in row.get("cases", []):
        shape = case.get("shape_mnk")
        if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
            continue
        clean["cases"].append({"shape_mnk": shape, "passed": case.get("passed") is True,
                               "output_tiles": number(case.get("output_tiles")), "k_chunks": number(case.get("k_chunks")),
                               "max_abs_error": number(case.get("max_abs_error"))})
    return clean


def export_bundle(report, results):
    """Pure conversion. Unknown text and nested fields are excluded by construction."""
    bundle = {"schema_version": 1, "task": "matmul-add", "task_contract": task_contract("matmul-add"),
              "task_contract_sha256": contract_digest("matmul-add"),
              "prompt_sha256": digest(report.get("prompt_sha256")),
              "knowledge_sha256": digest((report.get("knowledge") or {}).get("sha256")),
              "knowledge_enabled": (report.get("knowledge") or {}).get("enabled") is True if "knowledge" in report else None,
              "historical_knowledge_metadata_unavailable": "knowledge" not in report,
              "arms": [], "preflight_usage": usage_totals([request_usage(x) for x in report.get("provider_preflight", {}).get("requests", [])])}
    bundle["limits_per_arm"] = {key: number(report.get("limits_per_arm", {}).get(key)) for key in ("usd", "revisions", "requests", "seconds", "max_output_tokens")}
    implementation = report.get("implementation", {})
    revision = implementation.get("git_revision")
    bundle["implementation"] = {"git_revision": revision if isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40}", revision) else None,
                                "implementation_sha256": digest(implementation.get("implementation_sha256"))}
    for field in ("budget_at_start_usd", "budget_bounds_usd", "phase_cost_bounds_usd"):
        bundle[field] = {backend: number(report.get(field, {}).get(backend)) for backend in ("rsi", "pi")}
    for result in results:
        requests = [request_usage(x) for x in result["requests"]]
        generation = result.get("generation", {})
        arm = {"backend": choice(result.get("backend"), {"rsi", "pi"}),
               "provider": choice(result.get("provider"), {"openai", "deepseek"}),
               "model": choice(result.get("model"), {"deepseek-flash", "gpt-5.6-sol"}),
               "api": choice(result.get("api"), {"responses", "chat-completions"}),
               "status": choice(result.get("status"), STATUSES), "stop_reason": choice(result.get("stop_reason"), STOPS),
               "pi_version": choice(result.get("pi_version"), {"1.3.0"}) if result.get("backend") == "pi" else None,
               "pi_workflow": choice(result.get("pi_workflow"), {"persistent", "efficient", "native"}) if result.get("backend") == "pi" else None,
               "generation": {"max_tokens": number(generation.get("max_tokens")), "reasoning": choice(generation.get("reasoning"), {"high", "disabled"})},
               "prompt_sha256": digest(result.get("prompt_sha256")), "starter_sha256": digest(result.get("starter_sha256")),
               "final_source_sha256": digest(result.get("final_source_sha256")),
               "seconds": number(result.get("seconds")), "revisions": number(result.get("writes")),
               "requests": requests, "usage": usage_totals(requests),
               "validations": [validation(x) for x in result.get("validations", [])]}
        first = next((v for v in arm["validations"] if v["passed"]), None)
        arm["tokens_to_numerical_pass"] = first["cumulative_usage"] if first else None
        summary = next((a for a in report.get("arms", []) if a.get("backend") == arm["backend"]), {})
        arm["tiling_review"] = choice(summary.get("tiling_review"), {"passed", "pending", "not_applicable"})
        bundle["arms"].append(arm)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, help="local comparison name")
    parser.add_argument("--output", required=True, type=Path, help="new destination directory")
    parser.add_argument("--include-passing-source", action="store_true", help="copy a passing kernel after digest/privacy checks; review it before publishing")
    args = parser.parse_args(argv)
    comparisons = (LOCAL / "comparisons").resolve()
    directory = (comparisons / args.comparison).resolve()
    if directory.parent != comparisons:
        parser.error("comparison must name a local comparison")
    report = json.loads((directory / "comparison.json").read_text())
    results, sources = [], {}
    runs = (LOCAL / "runs").resolve()
    for arm in report["arms"]:
        run = (runs / arm["run"]).resolve()
        if run.parent != runs:
            parser.error("run must belong to local runs")
        result = json.loads((run / "result.json").read_text())
        results.append(result)
        if args.include_passing_source and result["status"] == "passed":
            path = run / "kernel.cpp"
            if path.is_symlink() or path.stat().st_size > 65536:
                parser.error("source must be a regular file of at most 64 KiB")
            source = path.read_bytes()
            sha = hashlib.sha256(source).hexdigest()
            if sha != result["final_source_sha256"] or not any(v["passed"] and v["source_sha256"] == sha for v in result["validations"]):
                parser.error("passing source digest mismatch")
            if re.search(rb"sk-[A-Za-z0-9_-]{16,}|/(?:home|data2|workspace)/|https?://|(?:\d{1,3}\.){3}\d{1,3}", source):
                parser.error("source contains potential private data; inspect it before exporting")
            backend = choice(result["backend"], {"rsi", "pi"})
            if backend == "unknown":
                parser.error("unknown source backend")
            sources[backend + "-kernel.cpp"] = source
    bundle = export_bundle(report, results)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence.json").write_text(json.dumps(bundle, indent=2) + "\n")
    for name, source in sources.items():
        (args.output / name).write_bytes(source)
    print(f"Exported {len(results)} arms; inspect {args.output} before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
