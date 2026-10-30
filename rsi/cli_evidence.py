"""CLI extension: caller invokes register(subparsers), then evidence_handler(args).

No cli.py ownership required. Production uses configured existing DB plus pinned
BGE. Standalone ingestion requires explicit DB/cache/CAS/outcome-store paths.
Instance ingestion publishes a copied DB through checked config revisions and
retains existing ingestion receipt references; optional new negatives append.
"""
import argparse
import json
from pathlib import Path

from core.artifacts import ContentAddressedArtifacts
from rsi.evidence_io import export_task, ingest_to_store


def register(commands):
    ingest = commands.add_parser('kdb-add-repo', help='index committed repository code/docs')
    ingest.add_argument('source', type=Path)
    ingest.add_argument('--database', type=Path)
    ingest.add_argument('--database-kind', choices=('a3', 'generic'))
    ingest.add_argument('--embedding-cache', type=Path)
    ingest.add_argument('--collection', required=True)
    ingest.add_argument('--repository', required=True, help='credential-free repository identity')
    ingest.add_argument('--negative-records', type=Path,
                        help='JSON list snapshot from existing knowledge-agent outcome store')
    ingest.add_argument('--evidence-store', type=Path)
    ingest.add_argument('--expect-revision')
    ingest.set_defaults(evidence_handler=_ingest)
    export = commands.add_parser('evidence-export', help='export retained canonical task evidence')
    export.add_argument('task', type=Path)
    export.add_argument('destination', type=Path)
    export.set_defaults(evidence_handler=_export)


def _ingest(args):
    instance = getattr(args, 'instance', None)
    if instance is None and any(getattr(args, name) is None for name in
                                ('database', 'embedding_cache', 'negative_records', 'evidence_store')):
        raise ValueError('standalone ingestion requires database, embedding-cache, negative-records and evidence-store')
    negatives = json.loads(args.negative_records.read_text()) if args.negative_records else []
    if not isinstance(negatives, list):
        raise ValueError('negative outcome snapshot must be a JSON list')
    if instance is not None:
        if args.database_kind not in (None, 'generic'):
            raise ValueError('instance CPU services require generic KDB schema')
        if any(getattr(args, name) is not None for name in ('database', 'embedding_cache', 'evidence_store')):
            raise ValueError('instance ingestion uses configured KDB, embeddings and instance CAS')
        from rsi.instance_ingestion import ingest_instance
        return ingest_instance(instance, args.source, collection=args.collection,
            repository=args.repository, negative_records=negatives, expect_revision=args.expect_revision)
    if args.database_kind in (None, 'a3'):
        from benchmarks.a3kernels.embeddings import PinnedBGEEmbeddings
        from benchmarks.a3kernels.knowledge import KnowledgeDB
        database = KnowledgeDB.create(args.database, PinnedBGEEmbeddings(cache_dir=args.embedding_cache))
    else:
        from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
        from benchmarks.a5kernels.knowledge import KnowledgeDB
        database = KnowledgeDB(args.database, PinnedBGEEmbeddings(cache_dir=args.embedding_cache))
    with database:
        return ingest_to_store(database, args.source, ContentAddressedArtifacts(args.evidence_store),
            collection=args.collection, repository=args.repository, negative_records=negatives)


def _export(args):
    return export_task(args.task, args.destination)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m rsi.cli_evidence')
    parser.add_argument('--instance', type=Path)
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(argv)
    print(json.dumps(args.evidence_handler(args), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
