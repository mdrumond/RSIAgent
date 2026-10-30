"""CLI extension: caller invokes register(subparsers), then evidence_handler(args).

No cli.py ownership required. Production uses configured existing DB plus pinned
BGE. An explicit outcome-store JSON snapshot is required, including [] if empty.
The receipt retains every supplied negative record, without reinterpretation.
"""
import argparse
import json
from pathlib import Path

from core.artifacts import ContentAddressedArtifacts
from rsi.evidence_io import export_task, ingest_to_store


def register(commands):
    ingest = commands.add_parser('kdb-add-repo', help='index committed repository code/docs')
    ingest.add_argument('source', type=Path)
    ingest.add_argument('--database', type=Path, required=True)
    ingest.add_argument('--database-kind', choices=('a3', 'generic'), default='a3')
    ingest.add_argument('--embedding-cache', type=Path, required=True)
    ingest.add_argument('--collection', required=True)
    ingest.add_argument('--repository', required=True, help='credential-free repository identity')
    ingest.add_argument('--negative-records', type=Path, required=True,
                        help='JSON list snapshot from existing knowledge-agent outcome store')
    ingest.add_argument('--evidence-store', type=Path, required=True)
    ingest.set_defaults(evidence_handler=_ingest)
    export = commands.add_parser('evidence-export', help='export retained canonical task evidence')
    export.add_argument('task', type=Path)
    export.add_argument('destination', type=Path)
    export.set_defaults(evidence_handler=_export)


def _ingest(args):
    negatives = json.loads(args.negative_records.read_text())
    if not isinstance(negatives, list):
        raise ValueError('negative outcome snapshot must be a JSON list')
    if args.database_kind == 'a3':
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
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(argv)
    print(json.dumps(args.evidence_handler(args), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
