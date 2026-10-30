"""CLI skill editors and actual service-backed manual tuning requests."""
import json
from pathlib import Path
import re

from core.artifacts import ContentAddressedArtifacts
from . import jobs
from .configuration import _locations
from .skill_inputs import inspect_skills, update_skills


def register(commands):
    skills = commands.add_parser('skills', help='inspect/edit baselines or evaluate skills')
    actions = skills.add_subparsers(dest='skill_action', required=True)
    get = actions.add_parser('get')
    get.add_argument('name', nargs='?')
    selection = get.add_mutually_exclusive_group()
    selection.add_argument('--builtin', action='store_true')
    selection.add_argument('--cycle')
    export = actions.add_parser('export', help='deliver a reusable accepted baseline folder')
    selection = export.add_mutually_exclusive_group(required=True)
    selection.add_argument('--cycle')
    selection.add_argument('--builtin', action='store_true')
    export.add_argument('--destination', type=Path, required=True)
    update = actions.add_parser('set')
    update.add_argument('name')
    update.add_argument('--file', type=Path, required=True)
    update.add_argument('--expect-bundle')
    mapping = actions.add_parser('map').add_subparsers(dest='map_action', required=True)
    mapping.add_parser('inspect')
    update = mapping.add_parser('set')
    update.add_argument('--file', type=Path, required=True)
    update.add_argument('--expect-bundle')
    tune = actions.add_parser('tune')
    tune.add_argument('--project', required=True)
    tune.add_argument('--guidance', required=True)


def _project(instance, reference):
    root, _ = _locations(instance)
    if re.fullmatch(r'[0-9a-f]{64}', reference):
        packet = ContentAddressedArtifacts(root / 'artifacts').read_json(reference)
        path = Path(packet['root']).resolve(strict=True)
    else:
        parts = reference.split('/')
        if len(parts) != 2 or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', p) for p in parts):
            raise ValueError('project reference must be retained CAS hash or CYCLE/OPERATION')
        path = root / 'runs' / parts[0] / 'projects' / parts[1]
    relative = path.relative_to(root / 'runs')
    if len(relative.parts) != 3 or relative.parts[1] != 'projects':
        raise ValueError('project must belong to this instance cycle')
    admitted = json.loads((path / 'service-result.json').read_text())['reference']
    if re.fullmatch(r'[0-9a-f]{64}', reference) and reference != admitted:
        raise ValueError('project receipt is not the retained service result')
    packet = ContentAddressedArtifacts(root / 'artifacts').read_json(admitted)
    if Path(packet['root']).resolve() != path or packet['operation_id'] != path.name:
        raise ValueError('project receipt scope differs')
    cycle_id = relative.parts[0]
    mode = json.loads((root / 'runs' / cycle_id / 'service.json').read_text())['spec']['mode']
    if mode == 'phase1':
        raise ValueError('phase1 keeps skills fixed; use a phase2/inference project')
    return cycle_id, mode, admitted


def execute(args):
    if args.skill_action == 'get':
        return inspect_skills(args.instance, args.name, builtin=args.builtin)
    if args.skill_action == 'map' and args.map_action == 'inspect':
        result = inspect_skills(args.instance)
        return {k: result[k] for k in ('bundle_ref', 'configuration_revision', 'mapping')}
    if args.skill_action == 'tune':
        if not args.guidance.strip():
            raise ValueError('tuning guidance must not be blank')
        cycle, mode, ref = _project(args.instance, args.project)
        return jobs.start(args.instance, dict(action='tune', mode=mode, cycle_id=cycle,
            steering={'project_ref': ref, 'user_guidance': args.guidance}))
    values = dict(mapping=json.loads(args.file.read_text())) if args.skill_action == 'map' else dict(
        name=args.name, content=args.file.read_text())
    return update_skills(args.instance, expect_bundle=args.expect_bundle, **values)
