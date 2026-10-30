"""Registration for instance setup and dashboard lifecycle; shared CLI owns dispatch."""
from pathlib import Path


def register(commands, instance_actions):
    setup = instance_actions.add_parser('setup', help='install isolated pinned instance dependencies')
    setup.add_argument('--python', default=None)
    setup.add_argument('--node', type=Path)
    setup.add_argument('--no-roles', action='store_true')
    setup.add_argument('--embeddings', action='store_true')
    setup.add_argument('--offline', action='store_true')
    setup.add_argument('--pip-cache', type=Path)
    setup.add_argument('--npm-cache', type=Path)
    setup.add_argument('--timeout', type=float, default=900)
    setup.set_defaults(runtime_handler=execute)
    dashboard = commands.add_parser('dashboard', help='control the instance loopback dashboard')
    actions = dashboard.add_subparsers(dest='action', required=True)
    start = actions.add_parser('start')
    start.add_argument('--port', type=int, default=0)
    for name in ('start', 'stop', 'status'):
        parser = start if name == 'start' else actions.add_parser(name)
        parser.set_defaults(runtime_handler=execute)


def execute(args):
    if args.command == 'instance':
        from .instance_setup import setup
        return setup(args.instance, python=args.python, node=args.node,
                     roles=not args.no_roles, embeddings=args.embeddings,
                     offline=args.offline, pip_cache=args.pip_cache, npm_cache=args.npm_cache,
                     timeout=args.timeout)
    from . import dashboard_runtime
    return getattr(dashboard_runtime, args.action)(
        args.instance, **({'port': args.port} if args.action == 'start' else {}))
