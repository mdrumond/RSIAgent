"""Launch, resume, or report an adaptive A5 Phase-1 campaign.

One-project smoke example::

    python -m benchmarks.a5kernels.phase1_adaptive_cli run --max-parallel 1 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
from benchmarks.a5kernels.phase1_adaptive_production import (
    AdaptiveProductionPaths,
    build_adaptive_phase1,
)
from benchmarks.a5kernels.phase1_experiments import (
    A5BackendModel,
    KnowledgeMode,
    ProfilingGuidance,
    admit_programming_guide,
    build_cells,
)
from benchmarks.a5kernels.phase1_production import load_direct_environment
from benchmarks.a5kernels.production import _transformers_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "resume", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--initial-memory", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--cpl-skills-revision", required=True)
    parser.add_argument("--tla-root", type=Path, required=True)
    parser.add_argument("--catlass-source", required=True)
    parser.add_argument("--catlass-revision", required=True)
    parser.add_argument("--model", choices=[item.value for item in A5BackendModel], required=True)
    parser.add_argument("--knowledge", choices=[item.value for item in KnowledgeMode], required=True)
    parser.add_argument("--profiling", choices=[item.value for item in ProfilingGuidance], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--kdb", type=Path)
    parser.add_argument("--collection")
    parser.add_argument("--bge-cache", type=Path)
    parser.add_argument("--env-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    guide = admit_programming_guide(
        args.guide,
        cpl_skills_revision=args.cpl_skills_revision,
        catlass_revision=args.catlass_revision,
    )
    cell = next(
        item for item in build_cells(guide)
        if item.model.backend_model is A5BackendModel(args.model)
        and item.knowledge is KnowledgeMode(args.knowledge)
        and item.profiling is ProfilingGuidance(args.profiling)
    )
    embeddings = None
    if cell.knowledge is KnowledgeMode.WITH_KDB:
        if args.bge_cache is None:
            raise ValueError("KDB-on CLI requires --bge-cache")
        embeddings = PinnedBGEEmbeddings(
            cache_dir=_transformers_cache(args.bge_cache)
        )
    campaign = build_adaptive_phase1(
        cell=cell,
        paths=AdaptiveProductionPaths(
            args.root,
            args.tla_root,
            args.catlass_source,
            args.catlass_revision,
            args.device,
            args.kdb,
            args.collection,
        ),
        objective=args.objective,
        initial_memory=args.initial_memory.read_bytes(),
        environment=load_direct_environment(env_file=args.env_file),
        embeddings=embeddings,
        max_parallel=args.max_parallel,
    )
    result = getattr(campaign, args.action)()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
