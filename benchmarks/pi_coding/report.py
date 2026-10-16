"""Provider usage accounting, with explicit unknowns and no subset double counting."""


def usage_totals(records):
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
              "cached_input_tokens": 0, "reasoning_tokens": 0, "unknown_requests": 0,
              "unknown_breakdown_requests": 0}
    for row in records:
        usage = row.get("usage") or {}
        inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(inputs) is not int or type(outputs) is not int:
            totals["unknown_requests"] += 1
            continue
        totals["input_tokens"] += inputs
        totals["output_tokens"] += outputs
        totals["total_tokens"] += inputs + outputs
        inp = usage.get("prompt_tokens_details") or {}
        if not inp and type(usage.get("prompt_cache_hit_tokens")) is int:
            inp = {"cached_tokens": usage["prompt_cache_hit_tokens"]}
        out = usage.get("completion_tokens_details") or {}
        if "cached_tokens" not in inp or "reasoning_tokens" not in out:
            totals["unknown_breakdown_requests"] += 1
        totals["cached_input_tokens"] += inp.get("cached_tokens", 0) or 0
        totals["reasoning_tokens"] += out.get("reasoning_tokens", 0) or 0
    return totals


def summarize(result, budget):
    usage = usage_totals(result["requests"])
    first = next((v for v in result["validations"] if v["passed"]), None)
    return {"backend": result["backend"], "run": result["run"], "status": result["status"],
            "stop_reason": result["stop_reason"], "usage": usage,
            "tokens_to_numerical_pass": first.get("cumulative_usage") if first else None,
            "first_passing_source_sha256": first["source_sha256"] if first else None,
            "tiling_review": "pending" if first else "not_applicable",
            "requests": len(result["requests"]), "revisions": result["writes"],
            "validations": len(result["validations"]), "seconds": result["seconds"],
            "reserved_or_charged_bound_usd": sum(r["charged_bound_usd"] for r in budget["requests"])}


def markdown_report(report):
    lines = ["# Matmul-add comparison", "", f"Status: {report['status']}", "",
             "| Backend | Numerical result | Stop reason | Input | Output | Total | Revisions | Seconds | Cost bound USD |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in report.get("arms", []):
        u = arm["usage"]
        lines.append(f"| {arm['backend']} | {arm['status']} | {arm['stop_reason']} | {u['input_tokens']} | {u['output_tokens']} | {u['total_tokens']} | {arm['revisions']} | {arm['seconds']} | {arm['reserved_or_charged_bound_usd']:.6f} |")
    lines += ["", "Cached input and reasoning are subsets of input and output, respectively. Unknown usage is not zero; inspect comparison.json. Cost bounds include retained reservations and preflight; they are not invoices.",
              "", "A numerical pass requires source review for tiling before declaring task success. No latency target is evaluated. One pair does not establish average performance."]
    if report.get("budget_reused_from"):
        lines += ["", "Budget reused from: " + report["budget_reused_from"] + ". Table cost bounds are cumulative across that allocation; token counts cover this phase only."]
        if "phase_cost_bounds_usd" in report:
            lines += ["", "This phase cost bounds (USD): " + ", ".join(f"{name}={amount:.6f}" for name, amount in report["phase_cost_bounds_usd"].items())]
    if report.get("interruption_note"):
        lines += ["", report["interruption_note"]]
    if report.get("error"):
        lines += ["", "Preflight: " + report["error"]]
    return "\n".join(lines) + "\n"
