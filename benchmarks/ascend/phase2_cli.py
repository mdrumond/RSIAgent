"""Read historical Phase 2 results; new work uses the subject curriculum."""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from .phase2 import admit_native_snapshot, build_native_phase2


def native_runner(args):
    from .phase1_cli import load_native_config, native_transport
    from benchmarks.a5kernels.phase1_production import load_direct_environment
    from benchmarks.a5kernels.phase2_production import _knowledge_factory

    config = load_native_config(args.config, getattr(args, "worker_config", None))
    snapshot = admit_native_snapshot(args.phase1_root, args.initial_memory.read_bytes(), args.review, args.root)
    identity = snapshot.destination.source_identity
    for key in ("target", "arch", "execution_profile", "cpl_skills_revision"):
        if config[key] != identity[key]:
            raise ValueError(f"native runtime changed Phase1 {key}")
    if identity.get("environment_identity") != config.get("environment_identity"):
        raise ValueError("native runtime environment differs from reviewed Phase1")
    knowledge_factory = None
    if identity.get("kdb_collection"):
        from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
        from benchmarks.a5kernels.knowledge import KnowledgeDB
        from benchmarks.a5kernels.production import _transformers_cache
        if config.get("collection") != identity["kdb_collection"]:
            raise ValueError("native Phase2 KDB collection changed")
        embeddings = PinnedBGEEmbeddings(cache_dir=_transformers_cache(Path(config["bge_cache"])))
        database = KnowledgeDB.open_read_only(Path(config["kdb"]), embeddings)
        try:
            manifests = database.selected_manifests(config["collection"])
            fingerprint = (manifests[0].fingerprint if isinstance(config["collection"], str)
                           else {manifest.collection: manifest.fingerprint for manifest in manifests})
            if (not any(manifest.language == "ascend-c" or "ascend-c" in manifest.kernel_languages
                        for manifest in manifests) or fingerprint != identity["kdb_manifest"]):
                raise ValueError("native Phase2 KDB evidence changed")
        finally:
            database.close()
        knowledge_factory = _knowledge_factory(SimpleNamespace(kdb=Path(config["kdb"]), collection=config["collection"]), embeddings)
    return build_native_phase2(snapshot, Path(config["guide"]).read_bytes(), native_transport(config, args.root),
                              environment=load_direct_environment(None, env_file=args.env_file),
                              knowledge_factory=knowledge_factory)


def main(argv=None, *, runner_factory=native_runner, sleeper=time.sleep):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--target", choices=("a3", "a5"), default="a5")
    target, remaining = selector.parse_known_args(argv)
    if target.target == "a5":
        from run_a5_phase2 import main as a5_main
        return a5_main(remaining)
    parser = argparse.ArgumentParser(description=__doc__, parents=[selector])
    parser.add_argument("command", choices=("admit", "run", "report"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--initial-memory", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(remaining)
    if args.command != 'report':
        parser.error('Ascend Phase 2 execution and admission are retired; use run_ascend_subjects.py; historical report remains available')
    state = args.root / 'state.json'
    result = dict(state=json.loads(state.read_bytes()) if state.is_file() else None,
                  records=[json.loads(path.read_bytes()) for path in sorted(args.root.rglob('terminal.json'))])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
