"""Build a standalone trusted verifier and canonical Catlass ProjectConfig."""
from pathlib import Path
import hashlib
import json
import re

PROJECT_SECONDS = 7200.0
RECOVERY_EXTENSION_SECONDS = 7200.0
ABI_VERSION = 1


def case_identity(seed):
    payload = dict(abi_version=ABI_VERSION, operation=seed['operation'], cases=seed['cases'],
                   exercise=seed['exercise'], allowed_helpers=seed.get('allowed_helpers', []))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_verifier(destination, *, seed, runtime_identity, physical_device, outer_cpl_handle,
                    execution_inputs=('candidate.py',), generated_paths=('outputs', 'cache'), source=None):
    """Write outside Actor-editable source; profiles supply all machine paths.

    The generated Python file embeds cases and identity, so PythonExecution's
    existing single-file frozen verifier snapshot retains the full host oracle.
    No adjacent mutable JSON file or Actor-provided oracle is loaded at runtime.
    """
    if not re.fullmatch(r'remote:bz-a5:(?:job|session):[^\s]+', outer_cpl_handle):
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
    if set(runtime_identity) != required or not runtime_identity['consumer_files']:
        raise ValueError('complete discovered runtime and compiler identity required')
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
                    execution_runtime='catlass-dsl', direct_physical=False, case_hash=case_identity(seed),
                    exercise=exercise, allowed_helpers=seed.get('allowed_helpers', []),
                    runtime_identity=runtime_identity, execution_inputs=list(execution_inputs),
                    generated_paths=list(generated_paths))
    text = Path(__file__).with_name('catlass_verifier.py').read_text()
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
        'performance_seconds': None,
    })


def elapsed_budget():
    from rsi.contracts import BudgetLimits

    return BudgetLimits(design_iterations=None, implementation_iterations=None,
                        execution_tries=None, actor_seconds=PROJECT_SECONDS, wall_seconds=None)


def worker_environment(*, inherited, outer_cpl_handle, physical_device, cache_directory):
    """Return env for a dedicated host worker, never mutate controller/global env.

    Main spawns the existing canonical worker with this env inside the outer
    registry-activated CPL session. This is not a worker loop or process runner.
    Environment activation and target interpreter selection remain profile-owned.
    """
    if not re.fullmatch(r'remote:bz-a5:(?:job|session):[^\s]+', outer_cpl_handle):
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
