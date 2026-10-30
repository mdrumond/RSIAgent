"""Argument registration and structured requests for local owned jobs."""
import json
from pathlib import Path
import uuid

from . import jobs
from .contracts import ProjectSpec


def register(commands):
    for mode in ('phase1', 'phase2', 'inference'):
        command = commands.add_parser(mode, help=f'control an instance {mode} job')
        actions = command.add_subparsers(dest='action', required=True)
        start = actions.add_parser('start')
        start.add_argument('--cycle', default=None)
        _inheritance_flags(start)
        if mode == 'inference':
            start.add_argument('--project', type=Path, required=True)
            start.add_argument('--no-recovery', action='store_true')
        for action in ('stop', 'resume', 'revisit', 'adopt'):
            entry = actions.add_parser(action)
            entry.add_argument('--job', required=True)
            if action == 'revisit':
                entry.add_argument('--cycle', required=True)
                _inheritance_flags(entry)
    command = commands.add_parser('job', help='inspect retained job state and evidence')
    actions = command.add_subparsers(dest='action', required=True)
    actions.add_parser('list')
    for action in ('status', 'evidence'):
        entry = actions.add_parser(action)
        entry.add_argument('job_id')


def _inheritance_flags(parser):
    parser.add_argument('--from-cycle')
    parser.add_argument('--inheritance', choices=('auto', 'parent', 'baseline'), default=None)
    parser.add_argument('--input-override', dest='input_overrides', action='append',
                        choices=('memory', 'skills', 'kdb'), default=[])


def _inheritance_request(args, previous_cycle=None):
    parent, policy = args.from_cycle, args.inheritance
    if args.action == 'revisit':
        policy = policy or 'parent'
        if parent is None and policy == 'parent':
            parent = previous_cycle
    elif policy is None:
        policy = 'baseline' if args.command == 'phase1' and parent is None else 'auto'
    if parent is not None and policy == 'baseline':
        raise ValueError('baseline inheritance cannot select a parent cycle')
    return dict(from_cycle=parent, inheritance=policy,
                input_overrides=list(dict.fromkeys(args.input_overrides)))


def execute(args):
    if args.command == 'job':
        return jobs.list_jobs(args.instance) if args.action == 'list' else getattr(
            jobs, args.action)(args.instance, args.job_id)
    if args.action == 'start':
        request = {'action': 'start', 'mode': args.command,
                   'cycle_id': args.cycle or str(uuid.uuid4())}
        if args.command == 'inference':
            request['projects'] = [ProjectSpec.from_dict(
                json.loads(args.project.read_text())).to_dict()]
            request['recover'] = not args.no_recovery
    else:
        previous = jobs.status(args.instance, args.job)
        if previous['request']['mode'] != args.command:
            raise ValueError('job belongs to another phase')
        if args.action == 'stop':
            return jobs.stop(args.instance, args.job)
        request = dict(previous['request'], action=args.action, previous_job=args.job)
        if args.action == 'resume' and previous['request'].get(
                'origin_action', previous['request'].get('action')) == 'tune':
            request['origin_action'] = 'tune'
        if args.action == 'revisit':
            if args.cycle == request['cycle_id']:
                raise ValueError('revisit requires a new cycle identity')
            request['previous_cycle'] = request['cycle_id']
            request['cycle_id'] = args.cycle
    if args.action in ('start', 'revisit'):
        request.update(_inheritance_request(args, request.get('previous_cycle')))
    return jobs.start(args.instance, request)
