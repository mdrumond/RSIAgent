"""Host-only child construction from frozen case templates; never dispatch work."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from core.artifacts import canonical_bytes
from rsi.contracts import ProjectSpec
from rsi.project_binding import source_identity
from rsi.project_runner import ProjectConfig
from rsi.python_execution import atomic_json


class MaterializationRejected(ValueError):
    """An admissible diagnostic proposal cannot use this host capability."""


HOST_CAPABILITY_CATALOG = tuple(dict(template_id=family + '.case-subset', parameters=dict(
    case_indices=dict(type='array', items=dict(type='integer'), minItems=1, uniqueItems=True)))
    for family in ('cpu', 'triton')) + (dict(template_id='triton.shape-reduction', parameters=dict(
        shape=dict(type='array', items=dict(type='integer'), minItems=1, uniqueItems=False))),)


@dataclass(frozen=True)
class MaterializedChild:
    spec: ProjectSpec
    config: ProjectConfig
    provenance: dict


def _require(condition, message):
    if not condition:
        raise MaterializationRejected(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unpack(value):
    binding = value.get('binding') if isinstance(value, dict) and 'config' in value else None
    value = value['config'] if isinstance(value, dict) and 'config' in value else value
    return (value if isinstance(value, ProjectConfig) else ProjectConfig.from_dict(value)), binding


def _argument(config, flag, *, mode='test'):
    argv = config.commands[mode]
    _require(flag in argv and argv.index(flag) + 1 < len(argv), 'Frozen command needs ' + flag)
    path = Path(argv[argv.index(flag) + 1])
    return path.resolve() if path.is_absolute() else (Path(config.source)/path).resolve()


def _cases(config):
    path = _argument(config, '--cases')
    argv = config.commands['test']
    if '--cases-sha256' in argv:
        _require(_sha(path) == argv[argv.index('--cases-sha256') + 1], 'Parent cases changed')
    data = json.loads(path.read_bytes())
    _require(isinstance(data, dict) and isinstance(data.get('cases'), list), 'Frozen case list required')
    return path, data


def capability_catalog(parent_spec, parent_config):
    """JSON host capabilities for context.templates, with actual frozen bounds."""
    config, binding = _unpack(parent_config)
    try:
        _, cases = _cases(config)
    except MaterializationRejected:
        return []
    count = len(cases['cases'])
    triton = '--runtime-manifest' in config.commands['test']
    if not count or triton and (not binding or cases.get('operation') not in ('add', 'row_sum', 'strided_copy')):
        return []
    templates = []
    if count > 1:
        entry = deepcopy(HOST_CAPABILITY_CATALOG[int(triton)])
        entry['parameters']['case_indices'].update(maxItems=count - 1,
            items=dict(type='integer', enum=list(range(count))))
        templates.append(entry)
    if triton:
        for index, case in enumerate(cases['cases']):
            shape = case.get('shape')
            if (not isinstance(shape, list) or not shape
                    or not all(type(n) is int and n > 0 for n in shape) or max(shape) == 1):
                continue
            choices = sorted({1, *shape, *(1 << i for i in range(max(shape).bit_length()))})
            entry = deepcopy(HOST_CAPABILITY_CATALOG[2])
            entry.update(template_id=entry['template_id'] + '.case-' + str(index),
                         case_index=index, parent_shape=list(shape),
                         shape_constraints=dict(maximum_per_dimension=list(shape),
                             at_least_one_strictly_smaller=True))
            entry['parameters']['shape'].update(minItems=len(shape), maxItems=len(shape),
                items=dict(type='integer', enum=choices))
            templates.append(entry)
    return [dict(entry, parent_project_id=parent_spec.project_id) for entry in templates]


def _copy_source(config, destination, *, triton):
    source = Path(config.source).resolve()
    verifier = Path(config.verifier).resolve()
    def ignore(directory, names):
        ignored = []
        for name in names:
            path = Path(directory)/name
            relative = path.relative_to(source).as_posix()
            if (name in ('.git', '__pycache__', '.pytest_cache') or path.resolve() == verifier
                    or triton and relative == '.execution-inputs'
                    or any(relative == p or relative.startswith(p.rstrip('/') + '/')
                           for p in config.generated_paths)):
                ignored.append(name)
        return ignored
    shutil.copytree(source, destination, ignore=ignore)


def _triton_bindings(parent_spec, config, binding, child, source, cases, inputs, oracle, root):
    from rsi.triton_project_config import project_config
    from rsi.triton_worker_binding import case_identity, prepare_worker_bindings
    manifest = Path(binding['manifest'])
    _require(_sha(manifest) == binding['sha256'], 'Parent binding manifest changed')
    table = json.loads(manifest.read_bytes())
    _require(parent_spec.project_id in table['projects'], 'Parent binding is not admitted')
    parent = table['projects'][parent_spec.project_id]
    _require(case_identity(config) == parent['logical_case_identity'], 'Parent workload differs from binding')
    devices = {}
    def retain(path):
        path = Path(path)
        data = path.read_bytes()
        destination = inputs/(hashlib.sha256(data).hexdigest() + path.suffix)
        if not destination.exists():
            destination.write_bytes(data)
        return str(destination)
    for physical, variant in parent['variants'].items():
        for resource in variant['trusted_resources']:
            _require(_sha(resource['path']) == resource['sha256'], 'Parent trusted resource changed')
        previous = ProjectConfig.from_dict(variant['config'])
        _require(_sha(previous.verifier) == _sha(oracle), 'Parent variants change the independent oracle')
        _require(_sha(variant['runtime_manifest']) == variant['runtime_sha256'], 'Parent runtime changed')
        options = dict(runtime_manifest=retain(variant['runtime_manifest']),
            executor_identity=variant['executor_identity'], python=previous.commands['test'][0],
            metadata=variant['metadata'])
        if '--exercise-profile' in previous.commands['validation']:
            options['exercise_profile'] = retain(_argument(previous, '--exercise-profile', mode='validation'))
        devices[physical] = options
    _require(bool(devices), 'At least one admitted physical variant required')
    descriptor = prepare_worker_bindings([dict(spec=child, source=str(source), cases=str(cases))],
                                         devices, root/'binding-draft.json')
    value = json.loads(Path(descriptor['manifest']).read_bytes())
    variants = value['projects'][child.project_id]['variants']
    # Own all trusted bytes even if the builder's package moves before execution.
    for variant in variants.values():
        built = ProjectConfig.from_dict(variant['config'])
        _require(_sha(built.verifier) == _sha(oracle), 'Builder changed the independent oracle')
        mapping = {r['path']: retain(r['path']) for r in variant['trusted_resources']}
        commands = {name: [mapping.get(arg, arg) for arg in argv] for name, argv in built.commands.items()}
        built = replace(built, verifier=str(oracle), commands=commands)
        variant['config'] = asdict(built)
        variant['spec']['command'] = commands['test']
        variant['runtime_manifest'] = mapping[variant['runtime_manifest']]
        for resource in variant['trusted_resources']:
            resource['path'] = mapping[resource['path']]
    path = root/'bindings.json'
    atomic_json(path, value)
    selected = variants[sorted(variants)[0]]
    return ProjectSpec.from_dict(selected['spec']), ProjectConfig.from_dict(selected['config']), dict(
        manifest=str(path), sha256=_sha(path))


def _read_child(path):
    value = json.loads(path.read_bytes())
    config = ProjectConfig.from_dict(value['config'])
    proof = value['provenance']
    _require(source_identity(config.source, verifier=config.verifier,
        generated_paths=config.generated_paths) == proof['child_source_hash']
        and _sha(config.verifier) == proof['oracle_sha256']
        and _sha(value['spec']['correctness_ref']) == proof['child_cases_sha256'],
        'Admitted child inputs changed')
    if proof['binding']:
        _require(_sha(proof['binding']['manifest']) == proof['binding']['sha256'], 'Child binding changed')
    provenance = dict(value['provenance'], receipt=dict(path=str(path), sha256=_sha(path)))
    return MaterializedChild(ProjectSpec.from_dict(value['spec']), config, provenance)


def read_materialized_child(path, *, sha256):
    """Read the exact durable host receipt referenced by the child registry."""
    path = Path(path)
    _require(_sha(path) == sha256, 'Child admission receipt changed')
    return _read_child(path)


def materialize_child(parent_spec, parent_config, proposal, *, root, remaining, evidence_refs, operation_id):
    """Create one linked child, with setup separate from explicit future allowance.

    Triton parent_config is host-only {'config': ProjectConfig wire data,
    'binding': {'manifest': path, 'sha256': hash}}. Plain CPU configs are supported.
    Completed admissions are idempotent; partial directories are never overwritten.
    The returned receipt supplies the new config and child-only binding manifest
    to service integration, without changing global frozen project/profile tables.
    evidence_refs identify admitted current research payloads; callers must keep
    historical service-result envelopes/events scoped to their origin archives.
    """
    started = time.monotonic()
    config, binding = _unpack(parent_config)
    _require(parent_spec.command == tuple(config.commands['test']), 'Parent spec/config command mismatch')
    fields = ('name', 'goal', 'acceptance', 'template_id', 'parameters', 'parent_project_id')
    idea = ({key: proposal[key] for key in fields if key in proposal} if isinstance(proposal, dict)
            else {key: getattr(proposal, key) for key in fields if hasattr(proposal, key)})
    _require(all(isinstance(idea.get(k), str) and idea[k].strip()
                 for k in ('name', 'goal', 'acceptance', 'template_id')), 'Diagnostic identity/goal/acceptance required')
    _require(idea.get('parent_project_id') in (None, parent_spec.project_id), 'Selected parent differs')
    _require(isinstance(operation_id, str) and bool(operation_id.strip()), 'Durable operation ID required')
    _require(isinstance(evidence_refs, (list, tuple)) and bool(evidence_refs)
        and all(isinstance(ref, str) and len(ref) == 64 and all(c in '0123456789abcdef' for c in ref)
                for ref in evidence_refs), 'Retained research evidence references required')
    original_cases, case_data = _cases(config)
    catalog = capability_catalog(parent_spec, parent_config)
    template = next((t for t in catalog if t['template_id'] == idea['template_id']), None)
    _require(template is not None, 'Unsupported host template')
    parameters = idea.get('parameters')
    requested_shape = None
    if 'shape' in template['parameters']:
        _require(isinstance(parameters, dict) and set(parameters) == {'shape'}, 'Only shape is supported')
        requested_shape, old = parameters['shape'], template['parent_shape']
        allowed = template['parameters']['shape']['items']['enum']
        _require(isinstance(requested_shape, list) and len(requested_shape) == len(old)
            and all(type(n) is int and 0 < n <= maximum and n in allowed
                    for n, maximum in zip(requested_shape, old))
            and any(n < maximum for n, maximum in zip(requested_shape, old)),
            'Shape must strictly shrink a parent dimension within the advertised choices')
        indices = [template['case_index']]
    else:
        _require(isinstance(parameters, dict) and set(parameters) == {'case_indices'}, 'Only case_indices is supported')
        indices = parameters['case_indices']
        _require(isinstance(indices, list) and 0 < len(indices) < len(case_data['cases'])
            and all(type(i) is int and 0 <= i < len(case_data['cases']) for i in indices)
            and len(set(indices)) == len(indices), 'Choose a nonempty proper subset of parent cases')
    triton = idea['template_id'].startswith('triton.')
    root = Path(root).resolve()
    source = Path(config.source).resolve()
    _require(not root.is_relative_to(source) and not source.is_relative_to(root), 'Child root must be independent')
    parent_source = source_identity(source, verifier=config.verifier, generated_paths=config.generated_paths)
    identity = dict(parent=parent_spec.to_dict(), config=asdict(config), binding=binding, proposal=idea,
        operation_id=operation_id, evidence_refs=list(evidence_refs), source_hash=parent_source,
        cases_sha256=_sha(original_cases), oracle_sha256=_sha(config.verifier))
    request_hash = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    receipt = root/'admission.json'
    if root.exists():
        _require(receipt.is_file(), 'Incomplete materialization requires host observation, never overwrite')
        _require(json.loads(receipt.read_bytes())['request_hash'] == request_hash, 'Materialization identity reused')
        return _read_child(receipt)
    def allowance():
        seconds = remaining() if callable(remaining) else remaining
        _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                 'Positive finite remaining elapsed allocation required')
        return seconds
    allocated = allowance()
    root.mkdir(parents=True)
    project, inputs, oracle = root/'project', root/'inputs', root/'verify.py'
    inputs.mkdir()
    _copy_source(config, project, triton=triton)
    shutil.copyfile(config.verifier, oracle)
    child_id = parent_spec.project_id + '-diagnostic-' + hashlib.sha256(
        canonical_bytes([parent_spec.project_id, operation_id])).hexdigest()[:16]
    # Preserve the verifier's template-family metadata, independently of the new
    # logical child ID. Only the curriculum-selected workload cases change.
    case_family = case_data.get('project_id', parent_spec.project_id)
    selected = [deepcopy(case_data['cases'][i]) for i in indices]
    if requested_shape is not None:
        selected[0]['shape'] = list(requested_shape)
    case_data = dict(case_data, cases=selected)
    cases = inputs/'cases.json' if triton else project/('diagnostic-cases-' + child_id.rsplit('-', 1)[-1] + '.json')
    _require(not cases.exists(), 'Child case path would overwrite a source resource')
    atomic_json(cases, case_data)
    child = replace(parent_spec, project_id=child_id, goal=idea['goal'] + '\nAcceptance: ' + idea['acceptance'],
        starting_folder=str(project), correctness_ref=str(cases))
    child_binding = None
    if triton:
        child, child_config, child_binding = _triton_bindings(
            parent_spec, config, binding, child, project, cases, inputs, oracle, root)
    else:
        def remap(arg):
            if arg == str(original_cases):
                return cases.relative_to(project).as_posix()
            if arg == str(Path(config.verifier).resolve()):
                return '{verifier}'
            path = Path(arg)
            return path.relative_to(source).as_posix() if path.is_absolute() and path.is_relative_to(source) else arg
        commands = {name: [remap(arg) for arg in argv] for name, argv in config.commands.items()}
        for argv in commands.values():
            if '--cases' in argv:
                argv[argv.index('--cases') + 1] = cases.relative_to(project).as_posix()
            if '--cases-sha256' in argv:
                argv[argv.index('--cases-sha256') + 1] = _sha(cases)
        child_config = replace(config, source=str(project), verifier=str(oracle), commands=commands)
        child = replace(child, command=tuple(commands['test']))
    allocated = min(allocated, allowance())
    limits = replace(parent_spec.budgets, actor_seconds=allocated,
                     wall_seconds=min(parent_spec.budgets.wall_seconds, allocated)
                     if parent_spec.budgets.wall_seconds is not None else None)
    child = replace(child, budgets=limits)
    if child_binding:
        table = json.loads(Path(child_binding['manifest']).read_bytes())
        entry = table['projects'][child_id]
        entry['logical']['budgets'] = limits.to_dict()
        for variant in entry['variants'].values():
            variant['spec']['budgets'] = limits.to_dict()
        final = root/'admitted-bindings.json'
        atomic_json(final, table)
        child_binding = dict(manifest=str(final), sha256=_sha(final))
    _require(source_identity(source, verifier=config.verifier, generated_paths=config.generated_paths) == parent_source
        and _sha(original_cases) == identity['cases_sha256'] and _sha(oracle) == identity['oracle_sha256'],
        'Parent inputs changed during setup')
    provenance = dict(parent_project_id=parent_spec.project_id, child_project_id=child_id,
        template_id=idea['template_id'], case_family_id=case_family, parent_source_hash=parent_source,
        parent_cases_sha256=identity['cases_sha256'], child_cases_sha256=_sha(cases),
        oracle_sha256=_sha(oracle), case_indices=indices, parameters=deepcopy(parameters), evidence_refs=list(evidence_refs),
        child_source_hash=source_identity(project, verifier=oracle, generated_paths=config.generated_paths),
        operation_id=operation_id, binding=child_binding, allocated_actor_seconds=allocated,
        setup_seconds=time.monotonic() - started, setup_excluded_from_actor_time=True)
    atomic_json(receipt, dict(schema='rsi-materialized-child-v1', request_hash=request_hash,
        spec=child.to_dict(), config=asdict(child_config), provenance=provenance))
    return _read_child(receipt)
