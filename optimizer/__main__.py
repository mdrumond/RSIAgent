"""Inspection/submission CLI; execution is provided by the optimizer worker."""
import argparse
import json
from pathlib import Path

from .jobs import JobStore
from .memory import discover_sources, freeze_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    discover = commands.add_parser('discover')
    discover.add_argument('--memory-root', action='append', required=True)
    discover.add_argument('--snapshot-root', action='append', default=[])
    create = commands.add_parser('create')
    create.add_argument('--store', required=True)
    create.add_argument('--source', required=True)
    create.add_argument('--entrypoint', required=True)
    create.add_argument('--signature', required=True, help='JSON file containing signatures')
    create.add_argument('--memory-root', required=True)
    create.add_argument('--snapshot-root', action='append', default=[])
    create.add_argument('--name', default='')
    create.add_argument('--idempotency-key')
    create.add_argument('--max-seconds', type=int, default=7200)
    for command in ('inspect', 'events', 'stop', 'resume', 'list'):
        child = commands.add_parser(command)
        child.add_argument('--store', required=True)
        if command != 'list':
            child.add_argument('job_id')
        if command == 'resume':
            child.add_argument('--additional-attempts', type=int, default=0, help=argparse.SUPPRESS)
            child.add_argument('--additional-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'discover':
        result = discover_sources(args.memory_root, snapshot_roots=args.snapshot_root)
    else:
        store = JobStore(args.store)
        if args.command == 'create':
            result = store.create(Path(args.source).read_text(), args.entrypoint,
                json.loads(Path(args.signature).read_bytes()),
                freeze_source(args.memory_root, snapshot_roots=args.snapshot_root),
                name=args.name, idempotency_key=args.idempotency_key, max_seconds=args.max_seconds)
        elif args.command == 'resume':
            result = store.resume(args.job_id, additional_attempts=args.additional_attempts,
                                  additional_seconds=args.additional_seconds)
        elif args.command == 'inspect':
            result = store.get(args.job_id)
        elif args.command == 'list':
            result = store.list()
        else:
            result = getattr(store, args.command)(args.job_id)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
