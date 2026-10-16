"""Paired, bounded coding comparison: python -m benchmarks.pi_coding.compare."""

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from dotenv import dotenv_values
import requests

from .budget import Budget, BudgetExceeded, allocation_lock
from .config import DEFAULT_KNOWLEDGE, PAIR_LIMITS, compose_request, source_metadata, runtime_preflight
from .tasks import contract_digest
from .experiment import LOCAL, parse_args as experiment_args, run_experiment, starter_source
from .gateway import Gateway, PROVIDERS
from .remote import RemoteAgent, Validator
from .report import markdown_report, summarize, usage_totals


def provider_preflight(env_file, budget, directory, provider="openai"):
    """One small billed access check, included in the first arm's $5 allocation."""
    if env_file.stat().st_mode & 0o077:
        raise ValueError("provider environment file must have mode 0600")
    key = dotenv_values(env_file)[PROVIDERS[provider]["key"]]
    with Gateway(provider, key, budget, "comparison-preflight", max_calls=1,
                 max_output_tokens=64, records_path=directory / ("preflight-requests-" + uuid.uuid4().hex[:8] + ".json")) as gateway:
        responses = provider == "openai"
        payload = {"model": gateway.spec["model"], "input" if responses else "messages": [{"role": "user", "content": "Reply OK."}]}
        response = requests.post(gateway.url + ("/responses" if responses else "/chat/completions"),
                                 headers={"Authorization": "Bearer " + gateway.token}, json=payload, timeout=150)
        ok = response.status_code == 200
        if ok:
            body = response.json()
            ok = not body.get("error") and (body.get("status") in ("completed", "incomplete") if responses else bool(body.get("choices")))
    errors = [r.get("provider_error") for r in gateway.records if r.get("provider_error")]
    return {"passed": ok, "requests": gateway.records, "usage": usage_totals(gateway.records),
            "error": errors[0] if errors else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), help="default: deepseek; resume preserves its saved provider")
    parser.add_argument("--resume", help="resume a comparison blocked before generation, retaining its ledgers")
    parser.add_argument("--reuse-budget-from", help="fresh paired run charged to an existing comparison allocation")
    parser.add_argument("--prompt-file", type=Path, help="identical reference-enhanced request for both arms")
    knowledge = parser.add_mutually_exclusive_group()
    knowledge.add_argument("--knowledge-file", type=Path, help="starting knowledge shared by both arms")
    knowledge.add_argument("--no-knowledge", action="store_true", help="omit the bundled knowledge notes")
    parser.add_argument("--connection", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=LOCAL / "providers.env")
    parser.add_argument("--budget-usd", type=float, help="required for new runs; divided equally; maximum $10")
    parser.add_argument("prompt", nargs="?", default=None)
    args = parser.parse_args(argv)
    if args.resume and args.reuse_budget_from:
        parser.error("resume and reuse-budget-from are mutually exclusive")
    if args.prompt_file:
        if args.prompt is not None or args.prompt_file.stat().st_size > 16384:
            parser.error("use one prompt of at most 16 KiB")
        args.prompt = args.prompt_file.read_text()
    budget_directory = None
    if args.resume and (args.knowledge_file or args.no_knowledge):
        parser.error("resume preserves the saved knowledge and prompt")
    if args.budget_usd is None and not (args.resume or args.reuse_budget_from):
        parser.error("new comparisons require an explicit --budget-usd")
    if args.reuse_budget_from:
        budget_directory = (LOCAL / "comparisons" / args.reuse_budget_from).resolve()
        if budget_directory.parent != (LOCAL / "comparisons").resolve():
            parser.error("reuse-budget-from must name a comparison directory")
        previous = json.loads((budget_directory / "comparison.json").read_text())
        if previous["status"] in ("running", "preflight"):
            parser.error("previous comparison must finish before reusing its allocation")
        if args.budget_usd is None:
            args.budget_usd = previous["total_limit_usd"]
        if previous["total_limit_usd"] != args.budget_usd:
            parser.error("reuse must preserve the original budget")
        budget_directory = Path(previous.get("budget_ledger_directory", budget_directory)).resolve()
        if budget_directory.parent != (LOCAL / "comparisons").resolve():
            parser.error("saved ledger must belong to the local comparisons directory")
        for backend in ("rsi", "pi"):
            if not (budget_directory / (backend + "-budget.json")).is_file():
                parser.error("existing allocation is missing a ledger")
    if args.budget_usd is not None and not 0 < args.budget_usd <= 10:
        parser.error("total budget must be in (0,10] USD")
    if args.resume:
        directory = (LOCAL / "comparisons" / args.resume).resolve()
        if directory.parent != (LOCAL / "comparisons").resolve():
            parser.error("resume must name a comparison directory")
        report = json.loads((directory / "comparison.json").read_text())
        if report.get("arms") or report["status"] not in ("blocked", "error"):
            parser.error("resume is supported only for blocked preflight, before either generation run")
        if args.budget_usd is None:
            args.budget_usd = report["total_limit_usd"]
        budget_directory = Path(report.get("budget_ledger_directory", directory)).resolve()
        if budget_directory.parent != (LOCAL / "comparisons").resolve():
            parser.error("saved ledger must belong to the local comparisons directory")
        if args.budget_usd != report["total_limit_usd"] or (args.prompt is not None and args.prompt != report["prompt"]):
            parser.error("resume must preserve the original budget and prompt")
        if not report.get("board_preflight", {}).get("passed"):
            parser.error("resume requires a previously passed board preflight; inspect any retained jobs first")
        args.provider = args.provider or report.get("provider", "deepseek")
        args.prompt = report["prompt"]
        report.pop("error", None)
    else:
        args.provider = args.provider or "deepseek"
        args.prompt = args.prompt or "Write a tiled D = A @ B + C kernel. Handle every edge tile and reduction tail. Validate on A2."
        request = args.prompt
        knowledge_path = args.knowledge_file or DEFAULT_KNOWLEDGE
        if not args.no_knowledge and knowledge_path.stat().st_size > 16384:
            parser.error("knowledge file exceeds 16 KiB")
        knowledge_text = "" if args.no_knowledge else knowledge_path.read_text(encoding="utf-8")
        args.prompt = compose_request(request, knowledge_text)
        directory = LOCAL / "comparisons" / (args.provider + "-matmul-" + uuid.uuid4().hex[:12])
        report = {"comparison": directory.name, "total_limit_usd": args.budget_usd, "arms": [], "prompt": args.prompt,
                  "limits_per_arm": {"usd": args.budget_usd / 2, **PAIR_LIMITS},
                  "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
                  "knowledge": {"enabled": bool(knowledge_text), "sha256": hashlib.sha256(knowledge_text.encode()).hexdigest()},
                  "task_contract_sha256": contract_digest("matmul-add"),
                  "starter_sha256": hashlib.sha256(starter_source("matmul-add").encode()).hexdigest(),
                  "implementation": source_metadata()}
    if budget_directory and args.reuse_budget_from:
        report["budget_reused_from"] = args.reuse_budget_from
        report["budget_ledger_directory"] = str(budget_directory)
    report["prompt_sha256"] = hashlib.sha256(args.prompt.encode()).hexdigest()
    # Reuse prompt validation before allocating a comparison directory.
    experiment_args(["--backend", "rsi", "--provider", args.provider, "--connection", str(args.connection), args.prompt])
    try:
        with allocation_lock(budget_directory or directory):
            return execute_pair(args, directory, budget_directory, report)
    except BudgetExceeded as exc:
        parser.error(str(exc))


def execute_pair(args, directory, budget_directory, report):
    """Run preflights and sequential arms while the allocation lease is held."""
    directory.mkdir(parents=True, exist_ok=True)
    report.update(status="preflight", provider=args.provider, model=PROVIDERS[args.provider]["model"])
    def save():
        (directory / "comparison.json").write_text(json.dumps(report, indent=2))
        (directory / "report.md").write_text(markdown_report(report))
    save()
    print(f"Comparison: {directory}", flush=True)
    budgets = {backend: Budget((budget_directory or directory) / (backend + "-budget.json"), args.budget_usd / 2) for backend in ("rsi", "pi")}
    report["budget_at_start_usd"] = {name: sum(r["charged_bound_usd"] for r in b.snapshot()["requests"]) for name, b in budgets.items()}
    try:
        try:
            runtime_preflight()
        except ValueError as exc:
            report.update(status="blocked", error=str(exc))
            return 2
        connection = json.loads(args.connection.read_text())
        preflight_dir = directory / "board-preflight"
        preflight_dir.mkdir(exist_ok=True)
        connection_digest = hashlib.sha256(args.connection.read_bytes()).hexdigest()
        reuse_board = report.get("connection_sha256") == connection_digest and report.get("board_preflight", {}).get("passed")
        report["connection_sha256"] = connection_digest
        if reuse_board:
            # Recheck live transport/session; a previously terminal passing fixture
            # can be reused when only billing/provider access blocked the pair.
            RemoteAgent(connection, preflight_dir / "remote").ready()
            board = report["board_preflight"]
        else:
            fixture = starter_source().replace("  missing_compile_symbol_0();\n", "", 1)
            (preflight_dir / "kernel.cpp").write_text(fixture)
            board = Validator(preflight_dir, connection).validate()
            report["board_preflight"] = board
        if not board["passed"]:
            report.update(status="blocked", error="A2 fixture preflight failed: " + board.get("reason", board["status"]))
            return 2
        provider = provider_preflight(args.env_file, budgets["rsi"], directory, args.provider)
        if "provider_preflight" in report:
            report.setdefault("previous_provider_preflights", []).append(report["provider_preflight"])
        report["provider_preflight"] = provider
        if not provider["passed"]:
            error = provider["error"] or {}
            report.update(status="blocked", error=args.provider + " access check failed: " + error.get("message", "inspect preflight-requests.json"))
            return 2
        report["status"] = "running"
        save()
        for backend in ("rsi", "pi"):
            run_args = experiment_args([
                "--backend", backend, "--provider", args.provider, "--task", "matmul-add",
                "--pi-workflow", "persistent", "--api", "responses" if args.provider == "openai" else "chat-completions",
                "--connection", str(args.connection), "--env-file", str(args.env_file),
                "--budget-file", str(budgets[backend].path), "--budget-usd", str(args.budget_usd / 2),
                "--max-revisions", str(PAIR_LIMITS["revisions"]), "--max-requests", str(PAIR_LIMITS["requests"]),
                "--timeout-seconds", str(PAIR_LIMITS["seconds"]),
                "--max-output-tokens", str(PAIR_LIMITS["max_output_tokens"]), args.prompt,
            ])
            result = run_experiment(run_args)
            report["arms"].append(summarize(result, budgets[backend].snapshot()))
            save()
            provider_failed = any(r.get("provider_error") or r.get("status") == "transport_error" for r in result["requests"])
            if result["status"] == "infrastructure_unavailable" or provider_failed:
                report.update(status="blocked", error="Stopped paired run after infrastructure/provider failure; inspect saved run evidence.")
                return 2
        report["status"] = "numerical_passes_pending_tiling_review" if all(a["status"] == "passed" for a in report["arms"]) else "completed_with_failures"
        return 0 if all(a["status"] == "passed" for a in report["arms"]) else 2
    except Exception as exc:
        report.update(status="error", error=type(exc).__name__)
        return 2
    finally:
        report["budget_bounds_usd"] = {name: sum(r["charged_bound_usd"] for r in b.snapshot()["requests"]) for name, b in budgets.items()}
        report["phase_cost_bounds_usd"] = {name: value - report["budget_at_start_usd"][name] for name, value in report["budget_bounds_usd"].items()}
        save()
        print(json.dumps({"comparison": directory.name, "status": report["status"], "error": report.get("error"), "report": str(directory / "report.md")}))


if __name__ == "__main__":
    raise SystemExit(main())
