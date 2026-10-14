#!/usr/bin/env python3
"""Plan the Fireworks SFT matrix or emit a clearly synthetic plumbing demo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from enum import Enum
import json
from typing import Sequence

from benchmarks.a5kernels.sft_experiment import (
    StudyConfig,
    aggregate_matrix,
    build_matrix,
    synthetic_demo_metrics,
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "synthetic-demo"))
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--adapter-model-id", required=True)
    parser.add_argument("--memory-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = StudyConfig(
        base_model_id=args.base_model_id,
        adapter_model_id=args.adapter_model_id,
        frozen_memory_id=args.memory_id,
        seeds=(17, 42, 91),
        max_tokens=32768,
        max_tool_calls=24,
        temperature=0.2,
        top_p=0.95,
    )
    matrix = build_matrix(config)
    payload = (
        {"config": asdict(config), "cells": [asdict(cell) for cell in matrix.cells]}
        if args.command == "plan"
        else {
            "evidence_kind": "synthetic-demo-not-execution-evidence",
            "executed": False,
            **aggregate_matrix(
                synthetic_demo_metrics(matrix, task_id="synthetic-vector-add")
            ),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
