"""Launch, resume, or report an adaptive A5 Phase-1 campaign.

One-project smoke example::

    python -m benchmarks.a5kernels.phase1_adaptive_cli run --max-parallel 1 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
from benchmarks.a5kernels.phase1_adaptive import AdaptivePhase1Coordinator
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
from benchmarks.a5kernels.profiling_guidance import admit_profiling_guidance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "resume", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--initial-memory", type=Path, required=True)
    parser.add_argument("--guide", type=Path)
    parser.add_argument("--cpl-skills-revision")
    parser.add_argument("--tla-root", type=Path)
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--catlass-source")
    parser.add_argument("--catlass-revision")
    parser.add_argument("--model", choices=[item.value for item in A5BackendModel])
    parser.add_argument("--knowledge", choices=[item.value for item in KnowledgeMode])
    parser.add_argument("--profiling", choices=[item.value for item in ProfilingGuidance])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--project-target", type=int)
    parser.add_argument("--max-projects", type=int, help="Deprecated metadata; does not limit execution")
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--verify-memory-usefulness", action="store_true")
    parser.add_argument("--usefulness-pair-seconds", type=float, default=600)
    parser.add_argument("--profiling-guidance-skill", type=Path)
    parser.add_argument("--curriculum-kdb", action="store_true")
    parser.add_argument("--kdb", type=Path)
    parser.add_argument("--collection", action="append")
    parser.add_argument("--bge-cache", type=Path)
    parser.add_argument("--env-file", type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    builder=build_adaptive_phase1,
    reporter=AdaptivePhase1Coordinator.read_retained_report,
    environment_loader=load_direct_environment,
    embedding_factory=PinnedBGEEmbeddings,
) -> None:
    args = _parser().parse_args(argv)
    initial_memory = args.initial_memory.read_bytes()
    if args.action == "report":
        result = reporter(args.root, args.objective, initial_memory)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return
    from benchmarks.ascend.execution import apply_worker_config, execution_mode
    worker = apply_worker_config({}, args.worker_config) if args.worker_config else None
    from benchmarks.ascend.worker_bindings import apply_a5_worker_arguments
    apply_a5_worker_arguments(args, worker, source="catlass_source", revision="catlass_revision")
    required = {
        "--guide": args.guide,
        "--cpl-skills-revision": args.cpl_skills_revision,
        "--tla-root": args.tla_root,
        "--catlass-source": args.catlass_source,
        "--catlass-revision": args.catlass_revision,
        "--model": args.model,
        "--knowledge": args.knowledge,
        "--profiling": args.profiling,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("live actions require " + ", ".join(missing))
    guide = admit_programming_guide(
        args.guide,
        cpl_skills_revision=args.cpl_skills_revision,
        catlass_revision=args.catlass_revision,
    )
    profiling_guide = guidance_text = None
    if args.profiling_guidance_skill is not None:
        if args.profiling != ProfilingGuidance.NEW.value:
            raise ValueError("--profiling-guidance-skill requires --profiling new")
        profiling_guide, guidance_text = admit_profiling_guidance(
            args.profiling_guidance_skill, cpl_skills_revision=args.cpl_skills_revision,
        )
    cell = next(
        item for item in build_cells(guide, profiling_guide=profiling_guide)
        if item.model.backend_model is A5BackendModel(args.model)
        and item.knowledge is KnowledgeMode(args.knowledge)
        and item.profiling is ProfilingGuidance(args.profiling)
    )
    embeddings = None
    if cell.knowledge is KnowledgeMode.WITH_KDB:
        if args.bge_cache is None:
            raise ValueError("KDB-on CLI requires --bge-cache")
        embeddings = embedding_factory(
            cache_dir=_transformers_cache(args.bge_cache)
        )
    campaign = builder(
        cell=cell,
        paths=AdaptiveProductionPaths(
            args.root,
            args.tla_root,
            args.catlass_source,
            args.catlass_revision,
            args.device,
            args.kdb,
            (args.collection[0] if args.collection and len(args.collection) == 1
             else tuple(args.collection) if args.collection else None),
            curriculum_kdb=args.curriculum_kdb,
            worker_config=worker,
        ),
        objective=args.objective,
        initial_memory=initial_memory,
        guide_bytes=args.guide.read_bytes(),
        environment=environment_loader(env_file=args.env_file),
        embeddings=embeddings,
        max_parallel=args.max_parallel,
        project_target=args.project_target,
        max_seconds=args.max_seconds,
        usefulness_enabled=args.verify_memory_usefulness,
        usefulness_pair_seconds=args.usefulness_pair_seconds,
        **({"max_projects": args.max_projects} if args.max_projects is not None else {}),
        **({"profiling_guidance_text": guidance_text} if guidance_text is not None else {}),
    )
    result = getattr(campaign, args.action)()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
