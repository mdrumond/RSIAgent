#!/usr/bin/env python3
"""Archive historical A5 campaigns and emit a model-neutral evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from benchmarks.a5kernels.baseline_archive import archive_baselines, load_archive_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = archive_baselines(load_archive_spec(value), args.destination)
    summary = {
        "campaigns": len({item["campaign_id"] for item in manifest["inventory"]}),
        "files": len(manifest["inventory"]),
        "manifest_sha256": hashlib.sha256(
            (args.destination / "raw/manifest.json").read_bytes()
        ).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
