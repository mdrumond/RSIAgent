"""Fresh device health plus short-lived, identity-scoped known-good probe reuse."""
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.phase1_project_bz import ProjectInfrastructureError
from .runtime_allocation import permitted_devices, unsettled_operations

_HEALTH = r'''
import importlib.metadata,json,os,pathlib,shutil,subprocess,sys
soc_query=subprocess.run([sys.executable,'-c','import torch_npu; print(torch_npu.npu.get_soc_version())'],capture_output=True,text=True,timeout=15)
soc=int(soc_query.stdout.strip()) if soc_query.returncode==0 and soc_query.stdout.strip().isdigit() else None
# This adapter's native host driver implements the dav2201 runtime binding.
arch='dav-2201' if soc==255 else None
rows={}
for kind in ['health','ecc','memory','usages','proc-mem','board']:
 try:
  r=subprocess.run(['npu-smi','info','-t',kind,'-i',sys.argv[1],'-c','0'],capture_output=True,text=True,timeout=4)
  rows[kind]={'exit':r.returncode,'text':r.stdout[:10000]}
 except subprocess.TimeoutExpired: rows[kind]={'exit':None,'text':'timeout'}
compiler=shutil.which('bisheng')
r=subprocess.run([compiler,'--version'],capture_output=True,text=True,timeout=5) if compiler else None
versions={}
for name in ['torch','torch-npu']:
 try: versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: versions[name]=None
home=os.environ.get('ASCEND_HOME_PATH')
candidates=[]
for value in [compiler,home,os.environ.get('ASCEND_TOOLKIT_HOME')]:
 if value:
  path=pathlib.Path(value).resolve()
  candidates.extend([path,*path.parents])
query=next((p/'query_pkg_version.sh' for p in candidates if (p/'query_pkg_version.sh').is_file()),None)
packages=subprocess.run(['bash',str(query)],capture_output=True,text=True,timeout=8) if query else None
print(json.dumps({'queries':rows,'compiler':r.stdout if r and r.returncode==0 else None,'versions':versions,'toolkit':packages.stdout if packages and packages.returncode==0 else None,'soc_version':soc,'arch':arch}))
'''


def fresh_health(transport, root, *, allowed_devices=None, health_workspace=None):
    if (not isinstance(health_workspace, str) or not Path(health_workspace).is_absolute()
            or '..' in Path(health_workspace).parts):
        raise ProjectInfrastructureError('configure an existing absolute health_workspace for readonly device checks')
    registry_root = getattr(transport, 'registry_workspace', None)
    if registry_root is not None:
        try:
            relative = Path(health_workspace).relative_to(registry_root)
        except ValueError:
            raise ProjectInfrastructureError('health_workspace must be beneath the registered workspace') from None
        if relative == Path('.') or '..' in relative.parts:
            raise ProjectInfrastructureError('health_workspace must be an existing subdirectory, not the registry root')
    identity = transport.runtime_identity
    device = identity['physical_device']
    if not permitted_devices([device], allowed_devices):
        raise ProjectInfrastructureError('physical device is outside campaign permissions')
    result = transport.run_remote(['python', '-c', _HEALTH, str(device)],
                                  key='health/' + uuid.uuid4().hex,
                                  cwd=health_workspace)
    if result.returncode:
        raise ProjectInfrastructureError('bounded device health query failed')
    report = json.loads(result.stdout)
    path = Path(root) / 'device-health' / (uuid.uuid4().hex + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, canonical(dict(report, handle=result.job_handle, runtime=identity, epoch=time.time())))
    q = report['queries']
    if any(q.get(k, {}).get('exit') != 0 for k in ['health','ecc','memory','usages','proc-mem','board']):
        raise ProjectInfrastructureError('device health query unavailable')
    if not re.search(r'Health Status\s*:\s*OK', q['health']['text']):
        raise ProjectInfrastructureError('device is unhealthy')
    if re.search(r':\s*[1-9]\d*', q['ecc']['text']):
        raise ProjectInfrastructureError('device ECC faults present')
    if ('No process in device' not in q['proc-mem']['text']
            or any(int(n) for n in re.findall(r'Ai(?:core|vector) Usage Rate\(%\)\s*:\s*(\d+)', q['usages']['text']))):
        raise ProjectInfrastructureError('device is not idle')
    hbm = re.search(r'HBM Usage Rate\(%\)\s*:\s*(\d+)', q['usages']['text'])
    if not hbm or int(hbm.group(1)) >= 50 or not report.get('compiler'):
        raise ProjectInfrastructureError('device memory or compiler health unavailable')
    board=q['board']['text']
    chip=re.search(r'Chip Name\s*:\s*(\S+)',board)
    npu=re.search(r'NPU Name\s*:\s*(\S+)',board)
    product=chip.group(1)+'_'+npu.group(1) if chip and npu else None
    return dict({key: report.get(key) for key in ('compiler', 'versions', 'toolkit','soc_version','arch')},
                board=board, product=product), str(path.resolve())


def verify_dependencies(actual, pinned):
    """Fail closed on unavailable or changed pinned runtime/toolchain facts."""
    environment=pinned.get('environment_identity') or {}
    expected={key:pinned.get(key) for key in ('target','arch','compiler_version')}
    expected.update({key:environment.get(key) for key in ('torch','torch_npu','cann_version','soc_version')})
    if any(value is None or value=='' for value in expected.values()):
        raise ProjectInfrastructureError('environment admission requires pinned product/architecture/compiler/CANN/torch/torch_npu/SoC identities')
    def packages(text):
        return {key.strip():value.strip() for key,value in
                (line.split(':',1) for line in text.strip().splitlines() if ':' in line)}
    versions=actual.get('versions') or {}
    mismatches=[]
    for field,current,wanted in [('product',actual.get('product'),expected['target']),
            ('architecture',actual.get('arch'),expected['arch']),
            ('torch',versions.get('torch'),expected['torch']),
            ('torch_npu',versions.get('torch-npu'),expected['torch_npu']),
            ('SoC',actual.get('soc_version'),expected['soc_version'])]:
        if current != wanted:mismatches.append(field)
    if not actual.get('compiler') or not actual['compiler'].startswith(expected['compiler_version']):
        mismatches.append('compiler')
    toolkit=actual.get('toolkit')
    if not toolkit or (packages(toolkit) != packages(expected['cann_version'])
            if packages(expected['cann_version']) else toolkit.strip()!=expected['cann_version'].strip()):
        mismatches.append('CANN')
    if mismatches:
        raise ProjectInfrastructureError('discovered environment differs from pinned '+', '.join(mismatches))


class EnvironmentAdmission:
    def __init__(self, check, transports, root, *, cache, dependencies, clock=time.time):
        self.check, self.transports, self.root = check, tuple(transports), Path(root)
        self.cache, self.dependencies, self.clock = Path(cache), dependencies, clock
        self.keys = set()
        self.cache.mkdir(parents=True, exist_ok=True)

    def invalidate(self):
        routes = [transport.runtime_identity for transport in self.transports]
        for path in self.cache.glob('*.json'):
            cached = json.loads(path.read_bytes())
            if path.stem in self.keys or any(route in cached.get('routes', []) for route in routes):
                path.unlink(missing_ok=True)
        self.keys.clear()

    def __call__(self, remaining, *, force=False):
        from benchmarks.a3kernels.native_smoke import smoke_plan
        identities, health_paths = [], []
        try:
            if unsettled_operations(self.root):
                raise ProjectInfrastructureError('unsettled native work prevents environment admission')
            for transport in self.transports:
                # Fresh health runs even on a known-good probe cache hit.
                health, path = fresh_health(transport, self.root,
                    allowed_devices=self.dependencies.get('allowed_devices'),
                    health_workspace=self.dependencies.get('health_workspace'))
                verify_dependencies(health, self.dependencies)
                health_paths.append(path)
                identities.append(dict(runtime=transport.runtime_identity, health=health))
            probe = smoke_plan(target=self.check.config.target, arch=self.check.config.arch,
                               profile=self.check.config.execution_profile)
            key = hashlib.sha256(canonical(dict(workers=identities, dependencies=self.dependencies,
                                               probe=probe.source_fingerprint))).hexdigest()
            self.keys.add(key)
            path = self.cache / (key + '.json')
            cached = json.loads(path.read_bytes()) if path.exists() else None
            now = self.clock()
            evidence = Path(cached['probe']) if cached else None
            valid_evidence = (evidence is not None and evidence.is_file()
                and hashlib.sha256(evidence.read_bytes()).hexdigest() == cached.get('probe_sha256'))
            if not force and valid_evidence and 0 <= now - cached['verified_epoch'] < 900:
                report = dict(status='reused', probe=cached['probe'], health=health_paths,
                              verified_epoch=cached['verified_epoch'], epoch=now)
            else:
                self.check(1, remaining)
                probe_path = str(self.check.last_report.resolve())
                report = dict(status='passed', probe=probe_path, health=health_paths,
                              probe_sha256=hashlib.sha256(Path(probe_path).read_bytes()).hexdigest(),
                              routes=[transport.runtime_identity for transport in self.transports],
                              verified_epoch=self.clock())
                atomic(path, canonical(report))
            atomic(self.root / 'environment-admission.json', canonical(report))
        except Exception:
            self.invalidate()
            raise
