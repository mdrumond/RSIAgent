"""Trusted exercise gates, embedded in the standalone frozen Catlass verifier.

Receipts are constructed from commands executed here and files created in a
fresh host-owned directory. Actor-provided JSON never supplies a tool verdict.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import contextlib
import ctypes
import csv
import re
import tempfile


if 'sha' not in globals():
    from rsi.catlass_verifier import sha, case_inputs, reference


OWNED_CAPTURE_HELPER_SHA256 = '1d362c3c862c7d59569290d4c3b9068b867a354d67ca86a43e4bdb4453efde59'


class CapabilityUnavailable(RuntimeError):
    """Explicit capability-study outcome; never an exercise-validation pass."""


@contextlib.contextmanager
def launch_output(directory=None):
    """Capture actual launch/synchronize fds, excluding staging/compiler output."""
    captured = {}
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            for name, stream in [('stdout',stdout), ('stderr',stderr)]:
                stream.seek(0)
                content = stream.read()
                captured[name] = content.decode(errors='replace')[:8192]
                captured[name + '_truncated'] = len(content) > 8192
                if directory is not None:
                    path = directory / (name + '.log')
                    path.write_bytes(content)
                    captured[name + '_path'] = str(path)
                captured[name + '_sha256'] = hashlib.sha256(content).hexdigest()


def receipt_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('required trusted tool artifact absent or empty')
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


@contextlib.contextmanager
def retained_evidence(common, exercise):
    """Retain raw negative evidence and typed receipt on every exit path."""
    outputs = Path('outputs')
    outputs.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='capability-' + exercise + '-', dir=outputs))
    receipt = dict(common, exercise=exercise, outcome='pending')
    try:
        yield directory, receipt
    except BaseException as error:
        receipt.update(outcome='capability-study-unavailable' if isinstance(error, CapabilityUnavailable)
                       else 'capability-failure',
                       failure=dict(type=type(error).__name__, message=str(error)[:8192]),
                       exercise_validated=False)
        raise
    finally:
        # Preserve the raw directory even if archive creation itself fails.
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        try:
            archive = shutil.make_archive(str(directory) + '-raw', 'gztar', directory)
            receipt['archive'] = receipt_file(archive)
        except Exception as archive_error:
            receipt['archive_error'] = dict(type=type(archive_error).__name__, message=str(archive_error))
        (outputs / (directory.name + '-receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')


def persist_process(process, directory, prefix):
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stream in ('stdout', 'stderr'):
        (directory / (prefix + '.' + stream)).write_bytes(getattr(process, stream))
    (directory / (prefix + '.json')).write_text(json.dumps(dict(returncode=process.returncode)) + '\n')


def run_retained(argv, *, directory, prefix='command', env=None):
    """Stream tool output to owned files before checking completion or verdict."""
    if directory is None:
        return subprocess.run(argv, capture_output=True, env=env)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = (directory / (prefix + '.' + stream) for stream in ('stdout','stderr'))
    try:
        with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
            process = subprocess.run(argv, stdout=stdout, stderr=stderr, env=env)
    except BaseException as error:
        (directory / (prefix + '.failure.json')).write_text(json.dumps(dict(
            type=type(error).__name__, message=str(error)[:8192])) + '\n')
        raise
    # Fake-consumer fixtures may return bytes directly; real subprocess output
    # was already streamed to these files, including before any interruption.
    if process.stdout is None:
        process.stdout = stdout_path.read_bytes()
    if process.stderr is None:
        process.stderr = stderr_path.read_bytes()
    persist_process(process, directory, prefix)
    return process


def tool_command(spec, arguments, *, env=None, evidence_directory=None, prefix="command"):
    tool = Path(spec['path'])
    if not tool.is_file():
        if spec.get('allow_unavailable', False):
            raise CapabilityUnavailable('required tool is not installed: ' + tool.name)
        raise ValueError('required installed tool is missing: ' + tool.name)
    if sha(tool) != spec['sha256']:
        raise ValueError('trusted tool identity changed')
    process = run_retained([str(tool), *arguments], directory=evidence_directory, prefix=prefix, env=env)
    return process


def owned_command(spec, replacements, *, contract, env=None, evidence_directory=None):
    # All command templates originate in the frozen trusted profile, not source.
    arguments = [arg.format_map(replacements) for arg in spec['arguments']]
    probe = tool_command(spec, spec.get('probe_arguments', ['--help']), env=env, evidence_directory=evidence_directory, prefix='probe')
    if probe.returncode:
        # A failed invocation is not proof that a feature is unsupported.
        raise ValueError('tool discovery failed; preserve diagnostic, not unavailable')
    process = tool_command(spec, arguments, env=env, evidence_directory=evidence_directory)
    receipt = dict(tool_sha256=spec['sha256'], arguments=arguments,
                   returncode=process.returncode,
                   stdout_sha256=hashlib.sha256(process.stdout).hexdigest(),
                   stderr_sha256=hashlib.sha256(process.stderr).hexdigest())
    if process.returncode:
        raise ValueError('required tool command failed: ' + process.stderr.decode(errors='replace'))
    markers = [line.split('=', 1)[1] for line in process.stdout.decode(errors='replace').splitlines()
               if line.startswith('RSI_CATLASS_RESULT=')]
    if not markers:
        raise ValueError('tool run has no trusted child launch receipt')
    child = json.loads(markers[-1])
    if (child.get('source_hash') != os.environ['RSI_EXPECTED_SOURCE_HASH']
            or child.get('case_hash') != contract['case_hash']
            or child.get('identity', {}).get('source_revision') != contract['runtime_identity']['source_revision']
            or child.get('identity', {}).get('dependency_revision') != contract['runtime_identity']['dependency_revision']
            or child.get('identity', {}).get('cann_version') != contract['runtime_identity']['cann_version']
            or [row['case'] for row in child.get('cases', [])] != [contract['cases'][0]]
            or not child.get('correctness') or not child.get('cases')
            or any(not case['launched'] or not case['numeric_match'] for case in child['cases'])):
        raise ValueError('tool launch receipt is not a verified current-source execution')
    receipt['launch_receipt'] = child
    return process, receipt


def child_arguments(case_index=0, mode='test'):
    return [str(Path(__file__).resolve()), '--mode', mode, '--case-index', str(case_index)]


def compiler_repair_gate(contract, compile_original, candidate_hash):
    exercise = contract['exercise']
    original = exercise.get('original_source')
    if not original or hashlib.sha256(original.encode()).hexdigest() == candidate_hash:
        raise ValueError('compile repair requires distinct frozen original and corrected source')
    common = dict(corrected_sha256=candidate_hash, case_hash=contract.get('case_hash'),
                  runtime=contract.get('runtime_identity'), outer_cpl_handle=contract.get('outer_cpl_handle'))
    with retained_evidence(common, 'compiler-original') as (directory, retained):
        path = directory / 'candidate.py'
        path.write_text(original)
        failure = None
        with launch_output(directory / 'compiler-output') as captured:
            try:
                compile_original(path)
            except Exception as error:
                failure = error
        if failure is None:
            raise ValueError('original compile unexpectedly succeeded; no compiler repair demonstrated')
        text = ''.join((directory / 'compiler-output' / (stream + '.log')).read_text(errors='replace')
                       for stream in ('stdout', 'stderr')) + repr(failure)
        (directory / 'exception.txt').write_text(repr(failure))
        if not re.search(exercise['diagnostic_pattern'], text):
            raise ValueError('original failed outside the frozen compiler diagnostic') from failure
        retained.update(outcome='expected-compiler-failure-observed', original_sha256=sha(path),
                        original_diagnostic_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        original_output=captured, corrected_sha256=candidate_hash,
                        corrected_compiled=False)
    return retained


def ir_gate(contract, temporary):
    spec = contract['exercise']['capture_tool']
    capture = temporary / 'capture'
    tools = contract['runtime_identity']['compiler_paths']
    arguments = ['capture', '--toolset', 'catlass-dsl', '--output-dir', str(capture),
                 '--real-tool', 'TlaCompile=' + tools['TlaCompile'],
                 '--real-tool', 'hivmc-a5=' + tools['hivmc-a5'], '--',
                 sys.executable, *child_arguments(mode='compile'), '--capture-root', str(capture)]
    # The capture helper is a trusted Python resource, invoked via this runtime.
    if (spec['sha256'] != OWNED_CAPTURE_HELPER_SHA256
            or sha(spec['path']) != OWNED_CAPTURE_HELPER_SHA256):
        raise ValueError('IR capture requires the exact owned no-shim helper identity')
    if spec.get('no_absolute_shims') is not True:
        raise ValueError('IR capture requires owned path-wrapper-only helper; shared compiler shims forbidden')
    env = dict(os.environ, CATLASS_DSL_CACHE_DIR=str(temporary / 'fresh-ir-cache'),
               CPL_CAPTURE_NO_ABSOLUTE_SHIMS='1')
    process = run_retained([sys.executable, spec['path'], *arguments], directory=temporary, prefix='capture-helper', env=env)
    if process.returncode:
        raise ValueError('actual compiler capture failed: ' + process.stderr.decode(errors='replace'))
    manifest = json.loads((capture / 'capture-manifest.json').read_text())
    if manifest['reference_exit_code'] != 0 or any(not row['restored'] for row in manifest.get('absolute_tool_shims', {}).values()):
        raise ValueError('compiler capture failed or instrumentation was not restored')
    invocations = []
    for name in ('TlaCompile', 'hivmc-a5'):
        rows = [row for row in manifest['invocations'] if row['tool'] == name]
        if not rows:
            raise ValueError('required real compiler invocation missing: ' + name)
        row = rows[-1]
        if row['exit_code'] != 0:
            raise ValueError('captured compiler invocation failed')
        def artifact(value):
            path = Path(value)
            return receipt_file(path if path.is_absolute() else capture / path)
        invocations.append(dict(tool=name, input=artifact(row['input_copy']),
                                output=artifact(row['output_copy']), argv=row['argv']))
    tree = ast.dump(ast.parse(Path('candidate.py').read_text()), include_attributes=True)
    return dict(ast_sha256=hashlib.sha256(tree.encode()).hexdigest(),
                staged_ast=tree, capture_manifest=receipt_file(capture / 'capture-manifest.json'),
                invocations=invocations)


def msprof_gate(contract, temporary):
    spec = contract['exercise']['tools']['msprof']
    reports = temporary / 'msprof'
    replacements = dict(report=str(reports), python=sys.executable,
                        verifier=str(Path(__file__).resolve()),
                        physical_device=str(contract['physical_device']),
                        workload=__import__('shlex').join([sys.executable, *child_arguments()]))
    _, receipt = owned_command(spec, replacements, contract=contract, evidence_directory=temporary / "tool-command")
    selected = []
    for path in reports.rglob('*.csv'):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get(spec['kernel_column']) == spec['kernel_name']:
                    duration = float(row[spec['duration_column']])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError('invalid exact-kernel profiler duration')
                    selected.append(dict(report=receipt_file(path), kernel=spec['kernel_name'], duration=duration))
    if not selected:
        raise ValueError('msprof produced no report for the frozen exact kernel selector')
    return dict(command=receipt, case=contract['cases'][0], kernel=spec['kernel_name'],
                reports=selected, metric='BasicInfo', instrumented=True)


def diagnostic_tool_gate(contract, name, temporary):
    spec = contract['exercise']['tools'][name]
    report = temporary / name
    replacements = dict(report=str(report), python=sys.executable,
                        verifier=str(Path(__file__).resolve()),
                        workload=__import__('shlex').join([sys.executable, *child_arguments()]),
                        physical_device=str(contract['physical_device']))
    process, receipt = owned_command(spec, replacements, contract=contract, evidence_directory=temporary / (name + "-command"))
    # Version-matched patterns bind actual CLI diagnostics/breakpoint observations,
    # not a receipt supplied by the Actor. Empty success output is not sufficient.
    diagnostic = process.stdout.decode(errors='replace') + process.stderr.decode(errors='replace')
    if name == 'device-printf':
        device = receipt['launch_receipt']['cases'][0].get('device_output', {})
        diagnostic = device.get('stdout', '') + device.get('stderr', '')
    if not spec.get('required_patterns') or any(not re.search(pattern, diagnostic) for pattern in spec['required_patterns']):
        raise ValueError(name + ' did not produce the required diagnostic observations')
    report.mkdir(parents=True, exist_ok=True)
    (report / 'observed-command.log').write_text(diagnostic)
    if name == 'device-printf':
        matches = list(re.finditer(spec['numeric_pattern'], diagnostic))
        left, right = case_inputs(contract['cases'][0], contract['operation'])
        expected = reference(contract['operation'], left, right)[0]
        if not matches or any(float(match.group('value')) != expected for match in matches):
            raise ValueError('device numeric print differs from independent expected output')
    artifacts = [receipt_file(path) for path in report.rglob('*') if path.is_file()]
    if not artifacts:
        raise ValueError(name + ' produced no retained diagnostic artifacts')
    return dict(command=receipt, observations=diagnostic, artifacts=artifacts)


def capability_gate(contract, *, compile_original, candidate_hash, repair_receipt=None):
    exercise = contract['exercise']
    kind = exercise['kind']
    if kind not in {'language-conversion','conversion','compiler-repair','api-ast-ir','msprof-basic','msdebug-mssanitizer',
                    'printf-numeric','missing-feature','device-environment-5xxxx'}:
        raise ValueError('unregistered exercise; numeric-only validation is not admitted')
    common = dict(source_hash=os.environ['RSI_EXPECTED_SOURCE_HASH'], candidate_sha256=candidate_hash,
                  case_hash=contract['case_hash'], runtime=contract['runtime_identity'],
                  outer_cpl_handle=contract['outer_cpl_handle'], physical_device=contract['physical_device'])
    with retained_evidence(common, kind) as (temporary, retained):
        if kind == 'language-conversion':
            original = Path(exercise['source_path'])
            provenance = exercise['provenance']
            if sha(original) != provenance['sha256']:
                raise ValueError('frozen Ascend C translation source changed')
            if json.loads(Path(exercise['provenance_path']).read_text()) != provenance:
                raise ValueError('Ascend C source provenance changed')
            result = dict(source_language='ascend-c', target_language='catlass-dsl',
                          original_source=receipt_file(original), provenance=provenance,
                          translated_candidate_sha256=candidate_hash,
                          semantic_oracle='independent host elementwise addition')
        elif kind == 'conversion':
            tree = ast.parse(Path('candidate.py').read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'to']
            # Numeric round-to-half probes plus explicit device conversions are
            # both required; this is structural AST evidence, not prose matching.
            aliases = {target.id: ast.unparse(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
            dtypes = {aliases.get(ast.unparse(arg), ast.unparse(arg)) for node in calls for arg in node.args}
            if not {'tla.Float16', 'tla.Float32'} <= dtypes:
                raise ValueError('conversion exercise requires explicit FP16 and FP32 device casts')
            result = dict(device_conversion_ast=True)
        elif kind == 'compiler-repair':
            result = repair_receipt or compiler_repair_gate(contract, compile_original, candidate_hash)
        elif kind == 'api-ast-ir':
            result = ir_gate(contract, temporary)
        elif kind == 'msprof-basic':
            result = msprof_gate(contract, temporary)
        elif kind == 'msdebug-mssanitizer':
            result = {name: diagnostic_tool_gate(contract, name, temporary) for name in ('msdebug', 'msSanitizer')}
        elif kind == 'printf-numeric':
            calls = {ast.unparse(node.func) for node in ast.walk(ast.parse(Path('candidate.py').read_text()))
                     if isinstance(node, ast.Call)}
            if exercise['printf_api'] not in calls:
                raise ValueError('candidate lacks the frozen supported device-print API call')
            result = diagnostic_tool_gate(contract, 'device-printf', temporary)
        elif kind == 'missing-feature':
            import catlass.tla as tla
            attribute = exercise['missing_api']
            if hasattr(tla, attribute):
                raise ValueError('claimed missing API is present at the selected runtime')
            tree = ast.parse(Path('candidate.py').read_text())
            calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            if not {'tla.cmp', 'tla.where'} <= calls:
                raise ValueError('missing-feature exercise requires compare/select implementation')
            result = dict(absent_api=attribute, supported_replacement=['tla.cmp', 'tla.where'])
        elif kind == 'device-environment-5xxxx':
            # A simulated fixture is explicitly distinct from hardware evidence.
            spec = exercise['tools']['environment-diagnostic']
            process, command = owned_command(spec, dict(report=str(temporary / 'environment')), contract=contract, evidence_directory=temporary / 'environment-command')
            text = process.stdout.decode(errors='replace') + process.stderr.decode(errors='replace')
            if not re.search(exercise['diagnostic_pattern'], text):
                raise ValueError('version-specific environment diagnostic not observed')
            if exercise.get('simulation', True):
                raise CapabilityUnavailable('simulated environment classification only; no actual device failure evidence')
            result = dict(command=command, diagnostic=text, recovered_numeric=True)
        else:
            raise ValueError('unregistered exercise; numeric-only validation is not admitted')
        retained.update(outcome='required-capability-verified', exercise_validated=True, evidence=result)
    return retained
