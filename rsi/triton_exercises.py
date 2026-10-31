"""Trusted current-candidate exercise producers, loaded as a frozen resource.

All commands run inside the canonical command's admitted target process group.
The immutable profile freezes tool argv, never candidate-specific attestations.
Missing tools and missing studies cannot become passing validation evidence.
"""
import ast
import csv
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


class StudyFailure(RuntimeError):
    def __init__(self, status, message, facts=None):
        super().__init__(message)
        self.status, self.facts = status, facts or {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


def device_print_values(text):
    """Parse inline output and Ascend HiIPU marker/next-line numeric blocks.

    Native Vec indices and unrelated log numbers are not printed tensor values.
    """
    values, active = [], False
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    for line in text.splitlines():
        if 'RSI_DEVICE_NUMERIC' in line:
            line = line.split('RSI_DEVICE_NUMERIC', 1)[1].lstrip(': ')
            active = True
        elif not active:
            continue
        if not line.strip():
            continue
        if re.fullmatch(r'[\s\[\],+\-.\deE]+', line):
            values.extend(float(value) for value in re.findall(number, line))
        else:
            active = False
    return values


class Producer:
    def __init__(self, args, contract, context):
        self.args, self.contract, self.context = args, contract, context
        self.root = Path(args.artifacts).resolve() / 'exercises'
        self.root.mkdir()
        self.facts = {}
        if not args.exercise_profile:
            self.profile = {}
        else:
            if digest(args.exercise_profile) != args.exercise_profile_sha256:
                raise StudyFailure('failed', 'Frozen exercise profile changed')
            self.profile = json.loads(Path(args.exercise_profile).read_text())
        self.bindings = {key: context[key] for key in
                         ('source_sha256', 'cases_sha256', 'runtime_manifest_sha256')}
        self.bindings['compiler_target'] = context['compiler_target']

    def fact(self, names, value):
        for name in names:
            self.facts[name] = dict(value, **self.bindings)

    def child(self, directory, candidate=None):
        return self.args.native_command(directory, candidate)

    def run(self, argv, directory, *, expected_failure=False, environment=None, seconds=300):
        """Stream raw output to disk; inherit the outer owned process group."""
        directory.mkdir(parents=True, exist_ok=True)
        if not argv or any(not isinstance(x, str) or not x for x in argv):
            raise StudyFailure('failed', 'Frozen tool command must be a nonempty argv')
        if not math.isfinite(seconds) or seconds <= 0:
            raise StudyFailure('failed', 'Study elapsed allocation must be positive')
        deadline = os.environ.get('RSI_COMMAND_DEADLINE_MONOTONIC')
        if deadline:
            seconds = min(seconds, float(deadline)-time.monotonic())
        if seconds <= 0:
            raise StudyFailure('failed', 'Remaining elapsed allocation exhausted')
        receipt = dict(self.bindings, argv=argv, status='pending', started_monotonic=time.monotonic())
        write(directory/'command.json', receipt)
        with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
            try:
                process = subprocess.Popen(argv, stdout=out, stderr=err,
                                           env=dict(os.environ, **(environment or {})))
            except OSError as exc:
                status = 'unsupported' if isinstance(exc, FileNotFoundError) else 'failed'
                receipt.update(status=status, error=str(exc))
                write(directory/'command.json', receipt)
                raise StudyFailure(status, 'Required executable could not start', receipt) from exc
            try:
                code = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                receipt.update(status='timed_out', elapsed_seconds=time.monotonic()-receipt['started_monotonic'],
                               closure='Outer command must close its owned process group')
                write(directory/'command.json', receipt)
                raise StudyFailure('failed', 'Study exceeded its remaining elapsed allocation', receipt) from exc
        receipt.update(returncode=code, elapsed_seconds=time.monotonic()-receipt['started_monotonic'],
                       status='passed' if (code != 0 if expected_failure else code == 0) else 'failed')
        write(directory/'command.json', receipt)
        if receipt['status'] != 'passed':
            raise StudyFailure('failed', 'Study command returned an unexpected result', receipt)
        return receipt

    def child_result(self, directory, source):
        for path in directory.rglob('result.json'):
            value = json.loads(path.read_text())
            if (value.get('status') == 'passed' and value.get('source_sha256') == source
                    and value.get('cases_sha256') == self.args.cases_sha256
                    and value.get('runtime_manifest_sha256') == self.args.runtime_sha256):
                return value
        raise StudyFailure('failed', 'Tool did not produce a verified launch of the current candidate')

    def baseline(self, name, names, spec=None):
        spec = spec or self.contract.get(name)
        if not spec:
            raise StudyFailure('unconfigured', 'Frozen failing source fixture is required: '+name)
        path = Path(spec['project_path']).resolve()
        if digest(path) != spec['sha256']:
            raise StudyFailure('failed', 'Frozen baseline fixture changed')
        directory = self.root/name
        directory.mkdir()
        (directory/'frozen-failure.py').write_bytes(path.read_bytes())
        command = self.run(self.child(directory, path), directory, expected_failure=True)
        if command['returncode'] not in spec.get('returncodes', [1]):
            raise StudyFailure('failed', 'Baseline terminated without its expected compiler exit', command)
        stderr = '\n'.join(p.read_text(errors='replace') for p in directory.rglob('stderr.log'))
        if not spec.get('diagnostic') or spec['diagnostic'] not in stderr:
            raise StudyFailure('failed', 'Failure did not reproduce the frozen compiler diagnostic')
        before, after = path.read_text(), Path(self.args.candidate).read_text()
        (directory/'repair.diff').write_text(''.join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile='frozen-failure', tofile='current-candidate')))
        self.fact(names, {'status': 'passed', 'baseline_sha256': spec['sha256'],
                          'corrected_sha256': self.context['source_sha256'],
                          'raw_directory': str(directory.relative_to(self.root))})

    def conversion(self):
        ref = self.contract.get('ascendc_reference')
        if not ref or not ref.get('commit'):
            raise StudyFailure('unconfigured', 'Frozen Ascend C source provenance is required')
        source = Path(ref['project_path']).resolve()
        if digest(source) != ref['sha256']:
            raise StudyFailure('failed', 'Frozen Ascend C source changed')
        (self.root/'ascendc-reference.cpp').write_bytes(source.read_bytes())
        self.fact(['source_conversion'], {'status': 'passed', 'ascendc_reference': ref,
                   'verification': 'current Triton launch, frozen dtype/shape cases and independent CPU oracle'})

    def feature(self):
        names = ['capability_failure', 'supported_replacement']
        if self.contract.get('feature_probe'):
            return self.baseline('feature_probe', names)
        absent = next((name for name in ('gather', 'histogram', 'associative_scan', 'sort')
                       if self.context['observed_api'].get(name) is False), None)
        if absent is None:
            raise StudyFailure('not_applicable', 'The active runtime provides every frozen API probe; no missing feature is claimed')
        tree = ast.parse(Path(self.args.candidate).read_text())
        tree.body.append(ast.Import([ast.alias('triton.language', '_rsi_probe_language')]))
        kernel = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                       and any(isinstance(d, ast.Attribute) and d.attr == 'jit' for d in node.decorator_list)), None)
        if kernel is None:
            raise StudyFailure('unsupported', 'The candidate exposes no supported probe insertion point')
        kernel.body.insert(0, ast.Expr(ast.Call(ast.Attribute(ast.Name('_rsi_probe_language', ast.Load()), absent,
                                                            ast.Load()), [], [])))
        path = self.root/'missing-feature-probe.py'
        path.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
        self.baseline('feature_probe', names, {'project_path': str(path), 'sha256': digest(path),
                                              'diagnostic': absent, 'observed_api_present': False})

    def printf(self):
        if not self.context['observed_api']['device_print']:
            raise StudyFailure('unsupported', 'Active Triton runtime has no device_print API')
        directory = self.root/'printf'
        directory.mkdir()
        tree = ast.parse(Path(self.args.candidate).read_text())
        count = 0
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            if not any(isinstance(d, ast.Attribute) and d.attr == 'jit' for d in function.decorator_list):
                continue
            for node in ast.walk(function):
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'store'
                        and len(node.value.args) > 1):
                    call = ast.Expr(ast.Call(ast.Attribute(node.value.func.value, 'device_print', ast.Load()),
                                             [ast.Constant('RSI_DEVICE_NUMERIC'), node.value.args[1]], []))
                    for parent in ast.walk(function):
                        for _, field in ast.iter_fields(parent):
                            if isinstance(field, list) and node in field:
                                field.insert(field.index(node), call)
                                count += 1
                                break
                        else:
                            continue
                        break
        if not count:
            raise StudyFailure('unsupported', 'No supported Triton store observation point for printf study')
        instrumented = directory/'instrumented.py'
        instrumented.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
        self.run(self.child(directory, instrumented), directory,
                 environment={'TRITON_DEVICE_PRINT': '1'})
        child = self.child_result(directory, digest(instrumented))
        samples = [case['actual_sample'][0] for case in child['cases'] if case.get('actual_sample')]
        lines = '\n'.join(path.read_text(errors='replace') for path in directory.rglob('stdout.log'))
        observed = device_print_values(lines)
        if not observed or not samples or not any(math.isclose(x, samples[0], rel_tol=1e-3, abs_tol=1e-3) for x in observed):
            raise StudyFailure('failed', 'Device printf did not match the independently verified output sample')
        self.fact(['device_printf', 'numeric_diagnosis'], {'status': 'passed', 'observed_values': observed,
                   'verified_samples': samples, 'instrumented_sha256': digest(instrumented)})

    def tool(self, name, names):
        spec = self.profile.get('tools', {}).get(name)
        if not spec:
            raise StudyFailure('unconfigured', 'No frozen command profile for '+name)
        directory = self.root/name
        directory.mkdir(exist_ok=True)
        child = self.child(directory)
        workload = directory/'workload.py'
        workload.write_text('import pathlib, subprocess, sys\n'
                            'result = subprocess.run('+repr(child)+')\n'
                            'for path in pathlib.Path('+repr(str(directory/'application'))+').rglob("*.log"):\n'
                            '    if path.name in ("stdout.log", "stderr.log"):\n'
                            '        sys.stdout.buffer.write(path.read_bytes())\n'
                            'sys.stdout.flush()\nsys.exit(result.returncode)\n')
        values = dict(candidate=str(Path(self.args.candidate).resolve()), verifier=str(Path(self.args.verifier_path).resolve()),
                      python=sys.executable, output_dir=str(directory), workload=str(workload),
                      application=shlex.join(child), physical_device=self.args.physical_device,
                      logical_device='0', project=str(Path.cwd()))
        seconds = spec.get('seconds', 300)
        targets = spec.get('supported_compiler_targets')
        if targets is not None and self.context['compiler_target'] not in targets:
            raise StudyFailure('unsupported', 'Study profile does not support the observed compiler target')
        if spec.get('probe_argv'):
            self.run([value.format_map(values) for value in spec['probe_argv']], directory/'probe', seconds=seconds)
            capability = name+'_capability'
            if capability in names:
                self.fact([capability], {'status': 'passed', 'probe_directory': str(directory/'probe')})
        steps = spec.get('steps', [])
        if not steps or not any(any(token in value for token in ('{application}', '{workload}', '{candidate}'))
                                for step in steps for value in step['argv']):
            raise StudyFailure('unconfigured', 'Tool profile must execute the current candidate workload')
        for i, step in enumerate(steps):
            admission = self.args.health_probe(directory/('health-'+str(i)), require_idle=name == 'msprof')
            if name == 'msprof':
                write(directory/'isolation.json', admission)
            argv = [value.format_map(values) for value in step['argv']]
            env = {key: value.format_map(values) for key, value in step.get('environment', {}).items()}
            self.run(argv, directory/('step-'+str(i)), environment=env, seconds=seconds)
        self.child_result(directory/'application', self.context['source_sha256'])
        reports = []
        for pattern in spec.get('required_files', []):
            paths = [p for p in directory.glob(pattern) if p.is_file() and p.stat().st_size]
            if not paths:
                raise StudyFailure('failed', 'Tool did not emit its required raw report: '+pattern)
            reports.extend({'path': str(path.relative_to(self.root)), 'sha256': digest(path)} for path in paths)
        if not reports:
            raise StudyFailure('unconfigured', 'A tool study requires a raw-report contract')
        summary = self.check_reports(name, spec, directory, reports)
        write(directory/'study-summary.json', summary)
        self.fact(names, {'status': 'passed', 'tool': name, 'reports': reports, 'summary': summary})

    def check_reports(self, name, spec, directory, reports):
        """Validate real report content with version-matched frozen field names."""
        text = '\n'.join((self.root/r['path']).read_text(errors='replace') for r in reports)
        if name == 'msprof':
            if not all(spec.get(key) for key in ('kernel_column', 'duration_column', 'duration_unit')):
                raise StudyFailure('unconfigured', 'Profiler requires matched CSV fields, units and an actual isolation report')
            isolation = json.loads((directory/'isolation.json').read_text())
            measurements = isolation.get('measurements', {})
            if (str(isolation.get('physical_device')) != self.args.physical_device
                    or isolation.get('admitted') is not True or measurements.get('idle') is not True
                    or str(measurements.get('health', '')).upper() != 'OK'):
                raise StudyFailure('failed', 'Profiler lacks fresh healthy idle ownership evidence')
            kernels = {launch['kernel'] for case in self.context['cases'] for launch in case['launches']}
            measured = []
            for report in reports:
                path = self.root/report['path']
                if path.suffix.lower() != '.csv':
                    continue
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        if any(kernel in row.get(spec['kernel_column'], '') for kernel in kernels):
                            value = float(row[spec['duration_column']])
                            if math.isfinite(value) and value > 0:
                                measured.append(value)
            if not measured:
                raise StudyFailure('failed', 'Profiler reported no positive duration for the verified candidate kernel')
            return {'kernel_samples': measured, 'duration_unit': spec['duration_unit'], 'isolation': isolation}
        if name == 'msdebug':
            expression = spec.get('source_location_regex')
            if not expression:
                raise StudyFailure('unconfigured', 'Debugger requires a version-matched source-location parser')
            locations = re.findall(expression, text)
            if not locations:
                raise StudyFailure('failed', 'Debugger emitted no source location for the candidate study')
            return {'source_locations': locations}
        if name == 'mssanitizer':
            if not spec.get('summary_json') or not spec.get('errors_field'):
                raise StudyFailure('unconfigured', 'Sanitizer requires a trusted report-parser step and errors field')
            summary = json.loads((directory/spec['summary_json']).read_text())
            if spec['errors_field'] not in summary or summary[spec['errors_field']] not in (0, []):
                raise StudyFailure('failed', 'Sanitizer report is incomplete or reports errors')
            return summary
        if name == 'ir':
            required = {'ttir': False, 'ttadapter': False, 'last_pass': False}
            for report in reports:
                path = self.root/report['path']
                for stage in required:
                    if stage in path.name and 'module' in path.read_text(errors='replace'):
                        required[stage] = True
            if not all(required.values()):
                raise StudyFailure('failed', 'Compiler capture did not retain all required same-source MLIR stages')
            return required
        raise StudyFailure('unconfigured', 'No report semantics configured for study '+name)

    def ir(self):
        found = {}
        for case in self.context['cases']:
            for launch in case['launches']:
                for name, artifact in launch['artifacts'].items():
                    for required in ('ttir', 'ttadapter', 'last_pass_mlir'):
                        if required.replace('_mlir', '') in name:
                            path = Path(self.args.artifacts)/artifact['file']
                            if path.is_file() and path.stat().st_size:
                                found[required] = {'status': 'passed', 'path': str(path), 'sha256': digest(path)}
        for name, value in found.items():
            self.fact([name], value)
        missing = {'ttir', 'ttadapter', 'last_pass_mlir'}-found.keys()
        if missing:
            self.tool('ir', sorted(missing))

    def environment(self):
        spec = self.contract.get('historical_environment')
        if not spec:
            raise StudyFailure('unconfigured', 'A real frozen historical environment diagnostic is required')
        path = Path(spec['project_path']).resolve()
        if digest(path) != spec['sha256']:
            raise StudyFailure('failed', 'Historical environment input changed')
        (self.root/'historical-environment.log').write_bytes(path.read_bytes())
        admission = self.args.health_probe(self.root/'environment-health', require_idle=False)
        self.fact(['environment_identity', 'health_receipt'], {'status': 'passed', 'fresh_admission': admission,
                   'observed_versions': self.context['observed_versions'],
                   'runtime_device_name': self.context['runtime_device_name']})
        self.fact(['original_diagnostic', 'recovery_explanation'], {'status': 'passed', 'historical_input': spec,
                   'scope': 'historical diagnostic compared with fresh tool observations and current verified candidate; no live fault induced'})

    def produce(self):
        native = Path(self.args.artifacts)/'native-result.json'
        self.fact(['successful_compile', 'kernel_launch', 'numeric_verification', 'ast_audit'],
                  {'status': 'passed', 'current_verifier_result': str(native), 'sha256': digest(native)})
        self.fact(['api_capability'], {'status': 'passed', 'observed_api': self.context['observed_api']})
        wanted = set(self.contract.get('required_evidence', []))
        groups = [({'source_conversion'}, self.conversion),
                  ({'compile_failure', 'source_repair_diff'}, lambda: self.baseline('compiler_failure', ['compile_failure', 'source_repair_diff'])),
                  ({'capability_failure', 'supported_replacement'}, self.feature),
                  ({'device_printf', 'numeric_diagnosis'}, self.printf),
                  ({'ttir', 'ttadapter', 'last_pass_mlir'}, self.ir),
                  ({'msprof_capability', 'profile_raw', 'profile_parse', 'isolated_device'},
                   lambda: self.tool('msprof', ['msprof_capability', 'profile_raw', 'profile_parse', 'isolated_device'])),
                  ({'msdebug_capability', 'debugger_transcript', 'source_location'},
                   lambda: self.tool('msdebug', ['msdebug_capability', 'debugger_transcript', 'source_location'])),
                  ({'mssanitizer_capability', 'sanitizer_raw', 'sanitizer_summary'},
                   lambda: self.tool('mssanitizer', ['mssanitizer_capability', 'sanitizer_raw', 'sanitizer_summary'])),
                  ({'original_diagnostic', 'environment_identity', 'health_receipt', 'recovery_explanation'}, self.environment)]
        for names, run in groups:
            if wanted & names:
                try:
                    run()
                except StudyFailure as exc:
                    self.fact(sorted((wanted & names)-self.facts.keys()),
                              {'status': exc.status, 'error': str(exc), 'raw_failure': exc.facts})
                    if exc.facts.get('status') == 'timed_out':
                        partial = dict(self.bindings, facts=self.facts, closure='outer owned process group required')
                        write(self.root/'evidence.json', partial)
                        raise StudyFailure('failed', 'Timed-out study requires outer command closure before any further study', partial) from exc
                except (OSError, ValueError, KeyError) as exc:
                    self.fact(sorted((wanted & names)-self.facts.keys()),
                              {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        manifest = dict(self.bindings, facts=self.facts)
        write(self.root/'evidence.json', manifest)
        missing = {name: self.facts.get(name, {}).get('status', 'missing') for name in wanted
                   if self.facts.get(name, {}).get('status') != 'passed'}
        if missing:
            status = 'unsupported' if set(missing.values()) == {'unsupported'} else 'failed'
            raise StudyFailure(status, 'Exercise study did not pass: '+json.dumps(missing), manifest)
        return manifest


def produce(args, contract, context):
    return Producer(args, contract, context).produce()
