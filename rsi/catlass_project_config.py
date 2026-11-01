"""Build a standalone trusted verifier and canonical Catlass ProjectConfig."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re

PROJECT_SECONDS = 7200.0
RECOVERY_EXTENSION_SECONDS = 7200.0
ABI_VERSION = 1


def case_identity(seed):
    """Workload/oracle identity independent of device/tool argv and paths."""
    from core.artifacts import canonical_bytes
    tree=ast.parse(Path(__file__).with_name('catlass_verifier.py').read_text())
    oracle=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                           and node.name in ('case_inputs','reference')],type_ignores=[])
    payload=dict(abi_version=ABI_VERSION,operation=seed['operation'],cases=seed['cases'],
        input_dtype=seed.get('input_dtype','float32'),output_dtype=seed.get('output_dtype','float32'),
        capacity=seed.get('capacity',512),padding_multiple=seed.get('padding_multiple',64),
        atol=seed.get('atol',0.),rtol=seed.get('rtol',0.),oracle=ast.dump(oracle,include_attributes=False))
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def frozen_cases(project_id, configuration):
    """Read the trusted logical seed, not an Actor-provided path or CLI flag."""
    from rsi.curriculum_materialization import MaterializationRejected
    entry = configuration.get('projects', {}).get(project_id)
    if not entry:
        raise MaterializationRejected('Catlass parent has no registered logical seed')
    path = Path(entry['seed_path']).resolve()
    seed = json.loads(path.read_bytes())
    if (not isinstance(seed, dict) or not isinstance(seed.get('cases'), list)
            or case_identity(seed) != entry['case_hash']):
        raise MaterializationRejected('Catlass frozen seed differs from its registered identity')
    return path, seed


def subset_configuration(configuration, child_id, cases):
    """Scope the existing registered binder to one owned diagnostic seed."""
    from copy import deepcopy
    result = deepcopy(configuration)
    result['projects'] = {child_id: dict(seed_path=str(cases),
        case_hash=case_identity(json.loads(Path(cases).read_bytes())))}
    return result


def freeze_verifier(destination, *, seed, runtime_identity, physical_device, outer_cpl_handle,
                    execution_inputs=('candidate.py',), generated_paths=('outputs', 'cache'), source=None):
    """Write outside Actor-editable source; profiles supply all machine paths.

    The generated Python file embeds cases and identity, so PythonExecution's
    existing single-file frozen verifier snapshot retains the full host oracle.
    No adjacent mutable JSON file or Actor-provided oracle is loaded at runtime.
    """
    if not re.fullmatch(r'remote:[a-zA-Z0-9_.-]+:job:[^\s]+', outer_cpl_handle):
        raise ValueError('actual outer BZ-A5 CPL admission handle required')
    if seed['operation'] not in {'add', 'multiply', 'relu', 'convert'}:
        raise ValueError('unsupported Catlass numeric operation')
    if type(physical_device) is not int or physical_device < 0:
        raise ValueError('main must supply a freshly admitted physical device')
    if not seed['cases']:
        raise ValueError('frozen cases required')
    for case in seed['cases']:
        if (type(case['length']) is not int or not 1 <= case['length'] <= 512
                or type(case['block_count']) is not int or not 1 <= case['block_count'] <= 32
                or type(case['seed']) is not int):
            raise ValueError('case violates ABI v1 capacity or launch geometry')
    required = {'catlass_source', 'source_revision', 'dependency_gitlink', 'dependency_revision',
                'consumer_files', 'cann_version', 'cann_version_file', 'cann_version_contents',
                'pipeline_manifest', 'pipeline_manifest_sha256', 'catlass_module_sha256', 'compiler_paths'}
    if not required <= set(runtime_identity) or set(runtime_identity) - required - {'device_admission', 'device_health', 'artifact_export_attribute', 'target_python_paths'} or not runtime_identity['consumer_files']:
        raise ValueError('complete discovered runtime and compiler identity required')
    paths=runtime_identity.get('target_python_paths',[])
    if not isinstance(paths,list) or any(not isinstance(path,str) or not Path(path).is_absolute() for path in paths):
        raise ValueError('target_python_paths must be an explicit registry-derived absolute path list')
    kinds = {'language-conversion', 'conversion', 'compiler-repair', 'printf-numeric', 'api-ast-ir', 'missing-feature',
             'msprof-basic', 'msdebug-mssanitizer', 'device-environment-5xxxx'}
    exercise = seed['exercise']
    required_operation = {'conversion':'convert', 'api-ast-ir':'multiply', 'missing-feature':'relu'}
    if seed['operation'] != required_operation.get(exercise['kind'], 'add'):
        raise ValueError('exercise and independent numeric operation disagree')
    if exercise['kind'] not in kinds:
        raise ValueError('explicit registered exercise required; numeric-only acceptance forbidden')
    if exercise['kind'] == 'language-conversion':
        provenance = exercise.get('provenance', {})
        if not all(provenance.get(k) for k in ('repository', 'revision', 'path', 'symbol', 'sha256')):
            raise ValueError('language conversion requires exact pinned Ascend C provenance')
        if not all(exercise.get(k) for k in ('source_path', 'provenance_path')):
            raise ValueError('language conversion requires Actor-readable frozen source inputs')
    if exercise['kind'] == 'compiler-repair' and not all(exercise.get(k) for k in ('original_source', 'diagnostic_pattern')):
        raise ValueError('compile repair needs frozen original source and original diagnostic selector')
    if exercise['kind'] in {'printf-numeric', 'msprof-basic', 'msdebug-mssanitizer', 'device-environment-5xxxx'} and not exercise.get('tools'):
        raise ValueError('exercise needs version-matched trusted tool commands')
    if exercise['kind'] == 'api-ast-ir':
        from rsi.catlass_capabilities import OWNED_CAPTURE_HELPER_SHA256
        tool = exercise.get('capture_tool', {})
        if tool.get('sha256') != OWNED_CAPTURE_HELPER_SHA256:
            raise ValueError('IR exercise requires the pinned owned no-shim helper, not the installed helper')
        helper = Path(tool['path'])
        if not helper.is_absolute():
            if source is None:
                raise ValueError('source directory required to verify the actual project-local IR helper')
            helper = Path(source) / helper
        if not helper.is_file() or hashlib.sha256(helper.read_bytes()).hexdigest() != OWNED_CAPTURE_HELPER_SHA256:
            raise ValueError('actual IR helper does not implement the pinned owned no-shim capability')
    contract = dict(abi_version=ABI_VERSION, operation=seed['operation'], cases=seed['cases'],
                    physical_device=physical_device, outer_cpl_handle=outer_cpl_handle,
                    execution_runtime='catlass-dsl', direct_physical=False,
                    profile_capture=bool(exercise['kind']=='msprof-basic' and
                                         exercise['tools']['msprof'].get('device_mode')=='direct-physical'),
                    case_hash=case_identity(seed),
                    exercise=exercise, allowed_helpers=seed.get('allowed_helpers', []),
                    runtime_identity=runtime_identity, execution_inputs=list(execution_inputs),
                    generated_paths=list(generated_paths))
    # The shared dependency owns both live and standalone admission code.
    # Fail closed until that dependency exports its self-contained stdlib source.
    from rsi.device_health_probe import standalone_source, standalone_hash
    shared_admission = standalone_source()
    if not isinstance(shared_admission,str) or not shared_admission.strip():
        raise ValueError('shared device admission standalone source unavailable')
    exported_hash=hashlib.sha256(shared_admission.encode()).hexdigest()
    if exported_hash!=standalone_hash():
        raise ValueError('shared device probe changed during frozen export')
    contract['device_admission_source_sha256']=exported_hash
    admission=runtime_identity.get('device_admission',{})
    receipt_path=Path(admission['outer_receipt'])
    receipt=json.loads(receipt_path.read_text())
    npu=receipt.get('npu',{})
    if (receipt.get('handle')!=outer_cpl_handle or receipt.get('target')!=outer_cpl_handle.split(':')[1]
            or receipt.get('runtime')!='catlass-dsl' or admission.get('outer_runtime')!='catlass-dsl'
            or physical_device not in npu.get('devices',[]) or not npu.get('pool') or not npu.get('checked_at')):
        raise ValueError('actual outer CPL receipt does not declare this runtime/device')
    contract['outer_admission']=dict(path=str(receipt_path.resolve()),
        sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        handle=outer_cpl_handle,physical_device=physical_device,runtime='catlass-dsl')
    text = Path(__file__).with_name('catlass_verifier.py').read_text()
    text += '\n_shared_probe_namespace = {}\n'
    text += 'exec(' + repr(shared_admission) + ', _shared_probe_namespace)\n'
    text += "_shared_device_local_probe = _shared_probe_namespace['local_probe']\n"
    text += '\n' + Path(__file__).with_name('catlass_capabilities.py').read_text()
    # repr of JSON text is a Python literal, never shell interpolation.
    text += '\nFROZEN_CONTRACT = json.loads(' + repr(json.dumps(contract, sort_keys=True)) + ')\n'
    text += "if __name__ == '__main__':\n    main(FROZEN_CONTRACT)\n"
    destination = Path(destination)
    with destination.open('x') as output:
        output.write(text)
    return hashlib.sha256(text.encode()).hexdigest()


def project_config(*, source, verifier, target_python):
    from rsi.project_runner import ProjectConfig

    return ProjectConfig.from_dict({
        'source': str(source), 'verifier': str(verifier),
        'executor_identity': None,
        'commands': {mode: [target_python, '{verifier}', '--mode', mode]
                     for mode in ('compile', 'test', 'validation')},
        'editable_files': ['candidate.py'], 'generated_paths': ['outputs', 'cache'],
        'retained_paths': ['outputs'],
        'performance_seconds': None,
    })


def elapsed_budget():
    from rsi.contracts import BudgetLimits

    return BudgetLimits(design_iterations=None, implementation_iterations=None,
                        execution_tries=None, actor_seconds=PROJECT_SECONDS, wall_seconds=None)


def worker_environment(*, inherited, outer_cpl_handle, physical_device, cache_directory):
    """Return env for a dedicated host worker, never mutate controller/global env.

    Main spawns the existing canonical worker with this env inside the outer
    registry-activated CPL job. This is not a worker loop or process runner.
    Environment activation and target interpreter selection remain profile-owned.
    """
    if not re.fullmatch(r'remote:bz-a5:job:[^\s]+', outer_cpl_handle):
        raise ValueError('actual outer BZ-A5 CPL admission handle required')
    if type(physical_device) is not int or physical_device < 0:
        raise ValueError('freshly admitted physical device required')
    cache = Path(cache_directory)
    if not cache.is_absolute():
        raise ValueError('task-owned absolute cache directory required')
    result = dict(inherited)
    result.pop('RSI_EXPECTED_SOURCE_HASH', None)
    result.update(ASCEND_RT_VISIBLE_DEVICES=str(physical_device),
                  CATLASS_DSL_CACHE_DIR=str(cache),
                  RSI_OUTER_CPL_HANDLE=outer_cpl_handle,
                  RSI_CPL_RUNTIME='catlass-dsl')
    return result


def project_spec(*, project_id, goal, config):
    """Finalize launch argv from the canonical config, never staging placeholders."""
    from rsi.contracts import ProjectSpec

    if config.executor_identity is not None:
        raise ValueError('A5 on-host binding uses default local execution')
    return ProjectSpec(project_id=project_id, goal=goal,
                       command=tuple(config.commands['test']), budgets=elapsed_budget(),
                       starting_folder=config.source, correctness_ref=config.verifier)


def apply_worker_context(context, *, parent_pid):
    """Apply only inside the existing spawned child, before runner construction.

    This helper neither spawns a process nor constructs an executor. The generic
    wave factory must pass its actual parent PID and invoke it in that child.
    """
    if type(parent_pid) is not int or parent_pid <= 0 or os.getpid() == parent_pid or os.getppid() != parent_pid:
        raise ValueError('worker environment may only be applied in a spawned child')
    if set(context) != {'outer_cpl_handle', 'physical_device', 'cache_directory'}:
        raise ValueError('explicit local worker mask/cache/outer-handle context required')
    environment = worker_environment(inherited=os.environ, **context)
    os.environ.pop('RSI_EXPECTED_SOURCE_HASH', None)
    for name in ('ASCEND_RT_VISIBLE_DEVICES', 'CATLASS_DSL_CACHE_DIR',
                 'RSI_OUTER_CPL_HANDLE', 'RSI_CPL_RUNTIME'):
        os.environ[name] = environment[name]
    return dict(context, pid=os.getpid(), parent_pid=parent_pid, runtime='catlass-dsl',
                logical_device=0, executor_identity=None)


def device_guard(request, *, phase, metadata, configuration, remaining):
    """Thin Hubble adapter to Jason's built-in shared guard; no health logic."""
    from rsi.device_admission import guard

    placement=metadata.get('controller_placement')
    if placement not in ('local','on'):
        raise ValueError('Catlass local guard requires admitted on-target placement')
    adapted=dict(metadata,controller_placement='on',executor_identity=None)
    if (os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!=str(adapted['physical_device'])
            or os.environ.get('RSI_OUTER_CPL_HANDLE')!=adapted.get('outer_handle')
            or os.environ.get('RSI_CPL_RUNTIME')!=adapted.get('runtime')):
        raise ValueError('guard child environment differs from actual outer admission')
    options=dict(configuration)
    if phase=='admission' and getattr(request,'dispatch_id',None):
        options['probe_id']=request.dispatch_id
    return guard(request,phase=phase,metadata=adapted,configuration=options,remaining=remaining)



def bind_project(request, logical_admission, *, metadata, configuration, remaining):
    """Resolve this child's actual lease before Runner.create; never on resume."""
    from dataclasses import asdict, replace
    import shutil
    from rsi.contracts import ProjectSpec
    from rsi.project_runner import ProjectConfig
    from rsi.project_binding import admission_hash, source_identity, validate_resolution, PUBLIC_RUNTIME

    def check_remaining():
        value=remaining()
        if value is not None and value<=0:
            raise TimeoutError('lease binding elapsed allocation exhausted')
    check_remaining()
    root=Path(request.root)
    if request.reopen or root.exists():
        raise ValueError('binder runs once before Runner.create; resume uses stored resolution')
    owner=json.loads((root.parent/'worker.json').read_text())
    if owner['pid']!=os.getpid() or owner['dispatch_id']!=request.dispatch_id:
        raise ValueError('project binding belongs only to the owning dispatched child')
    if logical_admission['operation_id']!=request.project_id:
        raise ValueError('logical admission operation differs from dispatched project')
    spec=ProjectSpec.from_dict(logical_admission['spec'])
    config=ProjectConfig.from_dict(logical_admission['config'])
    physical=metadata['physical_device']
    if (type(physical) is not int or str(physical)!=request.device
            or metadata.get('controller_placement') not in ('local','on')
            or metadata.get('runtime')!='catlass-dsl' or config.executor_identity is not None):
        raise ValueError('actual local Catlass lease metadata required before binding')
    if (os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!=str(physical)
            or os.environ.get('RSI_OUTER_CPL_HANDLE')!=metadata['outer_handle']
            or os.environ.get('RSI_CPL_RUNTIME')!=metadata['runtime']):
        raise ValueError('binder must run in the selected child environment')
    if (tuple(config.editable_files)!=('candidate.py',)
            or tuple(config.retained_paths)!=('outputs',)):
        raise ValueError('logical config must declare candidate-only edits and outputs retention')
    project=configuration['projects'][spec.project_id]
    seed=json.loads(Path(project['seed_path']).read_text())
    logical_case=case_identity(seed)
    if logical_case!=project['case_hash']:
        raise ValueError('logical project cases changed before lease binding')
    original_source=Path(config.source).resolve()
    original_verifier=Path(config.verifier).resolve()
    logical_generated=tuple(dict.fromkeys((*config.generated_paths,*config.retained_paths)))
    logical_source=source_identity(original_source,verifier=original_verifier,
                                   generated_paths=logical_generated)
    directory=root.parent/'lease-inputs'
    directory.mkdir()  # Fail closed on a previous partial binding; no overwrite.
    resolved_source=directory/'source'
    generated=logical_generated
    def ignored(folder,names):
        ignored_names=[]
        for name in names:
            path=Path(folder)/name
            relative=path.relative_to(original_source).as_posix()
            if (name in ('.git','__pycache__','.pytest_cache') or path.resolve()==original_verifier
                    or any(relative==item or relative.startswith(item.rstrip('/')+'/') for item in generated)):
                ignored_names.append(name)
            elif path.is_symlink():
                raise ValueError('lease source inputs must not contain symlinks')
        check_remaining()
        return ignored_names
    shutil.copytree(original_source,resolved_source,ignore=ignored)
    verifier=directory/'verify.py'
    inputs=[path.relative_to(resolved_source).as_posix() for path in sorted(resolved_source.rglob('*')) if path.is_file()]
    freeze_verifier(verifier,seed=seed,runtime_identity=configuration['runtime_identity'],
        physical_device=physical,outer_cpl_handle=metadata['outer_handle'],source=resolved_source,
        execution_inputs=inputs,generated_paths=generated)
    base=project_config(source=resolved_source,verifier=verifier,target_python=configuration['target_python'])
    resolved_config=replace(config,source=str(resolved_source),verifier=str(verifier),executor_identity=None,
        commands=dict(config.commands,**base.commands))
    resolved_source_id=source_identity(resolved_source,verifier=verifier,generated_paths=generated)
    if logical_source!=resolved_source_id:
        raise ValueError('lease binding changed logical source contents')
    resolved_spec=dict(spec.to_dict(),command=list(resolved_config.commands['test']),
        starting_folder=str(resolved_source),correctness_ref=str(verifier))
    actual_runtime=metadata.get('runtime_identity',{})
    public_runtime={}
    for key in PUBLIC_RUNTIME:
        value=metadata.get(key,actual_runtime.get(key))
        if value is None or isinstance(value,(str,int,float,bool)):
            if key in metadata or key in actual_runtime:public_runtime[key]=value
    public_runtime.update(physical_device=physical,logical_device=0)
    evidence=dict(logical_admission_hash=admission_hash(logical_admission),
        logical_project_id=spec.project_id,physical_device=physical,
        logical_case_identity=logical_case,resolved_case_identity=case_identity(seed),
        logical_source_identity=logical_source,resolved_source_identity=resolved_source_id,
        public_runtime=public_runtime)
    check_remaining()
    resolution=dict(spec=resolved_spec,config=json.loads(json.dumps(asdict(resolved_config))),evidence=evidence)
    return validate_resolution(logical_admission,resolution,device=request.device,executor_identity=None)
