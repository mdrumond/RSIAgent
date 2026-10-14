#!/usr/bin/env python3
"""Validate or explicitly submit the fixed Fireworks DeepSeek V4 Flash SFT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from config.runtime_paths import resolve_env_file

from benchmarks.a5kernels.fireworks import (
    FIREWORKS_BASE_MODEL,
    FIREWORKS_CREDENTIAL_ENV,
    FireworksHTTPTransport,
    FireworksManagedSFT,
    FireworksSFTContract,
    validate_chat_jsonl,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("dataset", type=Path, nargs="?", help="existing OpenAI chat JSONL")
    result.add_argument("--account-id", default=os.environ.get("FIREWORKS_ACCOUNT_ID"))
    result.add_argument("--dataset-id")
    result.add_argument("--output-model-id", required=True)
    result.add_argument("--job-id", required=True)
    result.add_argument("--provenance", type=Path, required=True)
    result.add_argument(
        "--submit", action="store_true",
        help="create paid provider resources; also requires RSI_ALLOW_FIREWORKS_SFT=1",
    )
    result.add_argument(
        "--observe-existing", action="store_true",
        help="resume polling from provisional provenance without any POST request",
    )
    result.add_argument("--poll-seconds", type=float, default=30.0)
    return result


def main(argv=None) -> int:
    load_dotenv(resolve_env_file(), override=False)
    args = parser().parse_args(argv)
    contract = FireworksSFTContract()
    if args.submit and args.observe_existing:
        parser().error("--submit and --observe-existing are mutually exclusive")
    if args.observe_existing:
        if not args.account_id:
            parser().error("--account-id or FIREWORKS_ACCOUNT_ID is required")
        api_key = os.environ.get(FIREWORKS_CREDENTIAL_ENV, "")
        if not api_key:
            parser().error(f"{FIREWORKS_CREDENTIAL_ENV} is required")
        client = FireworksManagedSFT(
            FireworksHTTPTransport(api_key), contract=contract,
            poll_seconds=args.poll_seconds,
        )
        value = client.observe(
            account_id=args.account_id, job_id=args.job_id,
            output_model_id=args.output_model_id, provenance_path=args.provenance,
        )
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    if args.dataset is None or args.dataset_id is None:
        parser().error("dataset and --dataset-id are required unless observing")
    identity = validate_chat_jsonl(args.dataset)
    preview = {
        "mode": "submit" if args.submit else "dry-run",
        "base_model": FIREWORKS_BASE_MODEL,
        "dataset_path": str(identity.path),
        "dataset_sha256": identity.sha256,
        "dataset_bytes": identity.byte_count,
        "dataset_examples": identity.example_count,
        "contract": asdict(contract),
    }
    if not args.submit:
        print(json.dumps(preview, sort_keys=True, indent=2))
        return 0
    if not args.account_id:
        parser().error("--account-id or FIREWORKS_ACCOUNT_ID is required with --submit")
    api_key = os.environ.get(FIREWORKS_CREDENTIAL_ENV, "")
    if not api_key:
        parser().error(f"{FIREWORKS_CREDENTIAL_ENV} is required with --submit")
    client = FireworksManagedSFT(
        FireworksHTTPTransport(api_key), contract=contract,
        poll_seconds=args.poll_seconds,
    )
    result = client.run(
        args.dataset,
        account_id=args.account_id,
        dataset_id=args.dataset_id,
        output_model_id=args.output_model_id,
        job_id=args.job_id,
        submit=True,
        provenance_path=args.provenance,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
