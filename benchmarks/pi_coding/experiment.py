"""Run an opt-in, bounded coding smoke comparison: python -m benchmarks.pi_coding.experiment.

The RSI arm reuses its model route and strict action parser, with automatic host
validation after each source write. It is a coding-loop slice, not a full RSI
learning campaign. The Pi arm uses native tools and the same host validator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from dotenv import dotenv_values

from .budget import Budget
from .observations import compact_observation
from .gateway import Gateway, PROVIDERS
from .remote import Validator
from .tools import CandidateTools
from .tasks import TASK_IDS, contract_digest, matmul_starter, task_contract, task_description


from .config import ROOT, LOCAL, RULES, TASK, source_metadata
from .backends import pi_attempt, rsi_attempt, revision_prompt


def starter_source(task="vector-add"):
    task_contract(task)
    if task == "matmul-add":
        return matmul_starter()
    # Reuse the repository-owned fixture without altering its A3 protocol metadata.
    source = (ROOT / "benchmarks/a3kernels/ascendc_runtime/kernel.cpp").read_text()
    source = source.replace("rsi_a3kernels", "rsi_pi_smoke").replace("RSIA3VectorAdd", "RSIPiVectorAdd")
    return source.replace("  AscendC::TPipe pipe;", "  missing_compile_symbol_0();\n  AscendC::TPipe pipe;", 1)


def final_status(tools, validator, trace):
    digest = hashlib.sha256((tools.root / "kernel.cpp").read_bytes()).hexdigest()
    matching = [v for v in validator.results if v["source_sha256"] == digest]
    # A model can stop immediately after its final write. Independently check that
    # artifact when a validation cycle remains; never reuse an older revision's pass.
    if tools.writes and not matching and len(validator.results) < validator.max_validations:
        observation = tools.execute("validate_candidate", {})
        trace.append({"type": "host_final_validation", "result": observation})
        matching = [observation]
    return matching[-1]["status"] if matching else "unverified"


def run_experiment(args):
    env_path = Path(args.env_file)
    if env_path.stat().st_mode & 0o077:
        raise ValueError("provider environment file must have mode 0600")
    credentials = dotenv_values(env_path)
    connection = json.loads(Path(args.connection).read_text()) if args.connection else None
    run_dir = LOCAL / "runs" / (args.backend + "-" + args.provider + "-" + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    source = starter_source(args.task)
    (run_dir / "kernel.cpp").write_text(source)
    rules = RULES.replace("three", str(args.max_revisions)).replace("smoke task", "task")
    prompt = task_description(args.task) + "\n" + rules
    if args.prompt:
        prompt += "\nUser request (within the selected task contract):\n" + args.prompt
    prompt += "\nStarter source:\n" + source
    (run_dir / "prompt.txt").write_text(prompt)
    started = time.monotonic()
    validator = Validator(run_dir, connection, task=args.task, max_validations=args.max_revisions)
    tools = CandidateTools(run_dir, validator, deadline=started + args.timeout_seconds, max_revisions=args.max_revisions)
    budget = Budget(Path(args.budget_file), args.budget_usd)
    trace = []
    result = {"backend": args.backend, "provider": args.provider, "model": PROVIDERS[args.provider]["model"],
              "task": args.task, "task_contract": task_contract(args.task),
              "task_contract_sha256": contract_digest(args.task),
              "user_prompt": args.prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "scope": "coding-comparison" if args.max_revisions > 3 else "coding-smoke", "starter_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "status": "unverified", "pi_version": "1.3.0" if args.backend == "pi" else None,
              "pi_workflow": args.pi_workflow if args.backend == "pi" else None,
              "generation": {"max_tokens": args.max_output_tokens or PROVIDERS[args.provider]["max_tokens"],
                             "reasoning": "high" if args.provider == "openai" else "disabled"}}
    result["implementation"] = source_metadata()
    result["limits"] = {"revisions": args.max_revisions, "requests": args.max_requests, "seconds": args.timeout_seconds}
    result["api"] = "responses" if args.backend == "pi" and args.provider == "openai" else args.api
    print(f"Running {args.task}: {args.backend}/{args.provider}; artifacts: {run_dir.name}", file=sys.stderr, flush=True)
    with Gateway(args.provider, credentials[PROVIDERS[args.provider]["key"]], budget, run_dir.name,
                 deadline=tools.deadline,
                 max_calls=args.max_requests or (args.max_revisions if args.backend == "pi" and args.pi_workflow == "efficient" else 9),
                 max_output_tokens=args.max_output_tokens, records_path=run_dir / "requests.json") as gateway:
        def record_validation(observation):
            from .report import usage_totals
            observation["cumulative_usage"] = usage_totals(gateway.records)
            observation["provider_requests"] = gateway.calls
            observation["revision"] = tools.writes
            print(f"Revision {tools.writes}: {observation['status']}; cumulative generation tokens: {observation['cumulative_usage']['total_tokens']}", file=sys.stderr, flush=True)
            (run_dir / "validation.json").write_text(json.dumps(validator.results, indent=2))
        tools.on_validation = record_validation
        try:
            if args.backend == "pi":
                pi_attempt(args.provider, gateway, tools, prompt, run_dir, trace, workflow=args.pi_workflow)
            else:
                rsi_attempt(args.provider, gateway, tools, prompt, trace, api=args.api)
            result["status"] = final_status(tools, validator, trace)
        except Exception as exc:
            # Never serialize provider/client exception bodies, which may contain credentials.
            result.update(status="error", error_type=type(exc).__name__)
            if time.monotonic() >= tools.deadline:
                gateway.stop_reason = "deadline"
        result["stop_reason"] = gateway.stop_reason or (
            result["status"] if result["status"] in ("passed", "infrastructure_unavailable", "error")
            else "revision_limit" if tools.writes >= tools.max_revisions
            else "request_limit" if gateway.calls >= gateway.max_calls else "agent_stopped")
        if gateway.stop_reason and result["status"] != "passed":
            result["status"] = "limit_reached"
        result["requests"] = gateway.records
        result["request_limit"] = gateway.max_calls
    result.update(seconds=round(time.monotonic() - started, 3), writes=tools.writes,
                  truncated_responses=sum(e.get("type") in ("pi_revision_truncated", "rsi_revision_truncated") for e in trace),
                  validations=validator.results,
                  final_source_sha256=hashlib.sha256((run_dir / "kernel.cpp").read_bytes()).hexdigest())
    (run_dir / "trace.json").write_text(json.dumps(trace, indent=2))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"run": run_dir.name, "status": result["status"], "seconds": result["seconds"],
                      "requests": len(result["requests"]), "writes": result["writes"]}))
    result["run"] = run_dir.name
    return result


def run(args):
    return 0 if run_experiment(args)["status"] == "passed" else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rsi", "pi"), required=True)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), required=True)
    parser.add_argument("--env-file", default=str(LOCAL / "providers.env"))
    parser.add_argument("--budget-file", default=str(LOCAL / "budget.json"))
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--api", choices=("chat-completions", "responses"), default="chat-completions")
    parser.add_argument("--task", choices=TASK_IDS, default="vector-add")
    parser.add_argument("--pi-workflow", choices=("efficient", "native", "persistent"), default="efficient",
                        help="efficient: host-owned revisions; native: original tools; persistent: Pi-owned debugging session")
    parser.add_argument("prompt", nargs="?", help="implementation request for the selected task (quote it)")
    parser.add_argument("--prompt", dest="inline_prompt", help="alternative to the trailing prompt")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 file containing your implementation request")
    validation = parser.add_mutually_exclusive_group(required=True)
    validation.add_argument("--connection", help="ignored local A2 JSON configuration")
    validation.add_argument("--provider-smoke-only", action="store_true",
                            help="exercise coding/tools; explicitly report A2 validation unavailable")
    args = parser.parse_args(argv)
    if not 0 < args.budget_usd <= 10 or not 1 <= args.max_revisions <= 12 or not 1 <= args.timeout_seconds <= 3600:
        parser.error("limits must be budget (0,10], revisions [1,12], timeout [1,3600]")
    if args.max_requests is not None and not 1 <= args.max_requests <= 30:
        parser.error("max requests must be in [1,30]")
    if args.max_output_tokens is not None and not 1 <= args.max_output_tokens <= 16384:
        parser.error("max output tokens must be in [1,16384]")
    if args.api == "responses" and args.provider != "openai":
        parser.error("Responses requires OpenAI")
    if sum(value is not None for value in (args.prompt, args.inline_prompt, args.prompt_file)) > 1:
        parser.error("use only one trailing prompt, --prompt, or --prompt-file")
    if args.prompt_file is not None:
        if args.prompt_file.stat().st_size > 16384:
            parser.error("prompt file exceeds 16 KiB")
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.inline_prompt is not None:
        args.prompt = args.inline_prompt
    if args.prompt is not None and (not args.prompt.strip() or len(args.prompt.encode()) > 16384):
        parser.error("prompt must be nonempty text of at most 16 KiB")
    return args


def main():
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
