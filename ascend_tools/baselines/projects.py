"""Create independent editable starters and two-file frozen verifier bundles."""
import gzip
import io
import json
from pathlib import Path
import shlex
import tarfile
from . import describe, file_hash, validate_receipt


def prepare_project(workload,target,root,receipt,identity,python='python3',project_id=None):
    if target not in ('a3','a5'):raise ValueError('Unsupported campaign target')
    expected={**describe(workload),'target':target,'runtimeFingerprint':identity['fingerprint']}
    validate_receipt(receipt,expected)
    if receipt['runtime'] != identity:
        raise ValueError('Qualification identity differs from project runtime')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir();frozen=root/'frozen';frozen.mkdir()
    (source/'kernel.py').write_text('def run(*args, **attrs):\n    raise NotImplementedError("Implement the frozen workload on the configured NPU")\n')
    language='Triton-Ascend' if target=='a3' else 'Catlass DSL'
    contract={'language':language,'format':1,'workload':workload,'target':target,'expected':expected,
              'qualification':receipt,'identity':identity,'warmup':5,'repeats':20}
    package=Path(__file__).parent
    files={'ascend_tools/__init__.py':b'', 'contract.json':json.dumps(contract,sort_keys=True).encode()}
    for path in package.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files['ascend_tools/baselines/'+path.relative_to(package).as_posix()]=path.read_bytes()
    archive=frozen/'verifier.tar.gz'
    with archive.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444;tar.addfile(info,io.BytesIO(data))
    launcher=frozen/'verify.py'
    launcher.write_text(_LAUNCHER.replace('ARCHIVE_HASH',file_hash(archive)))
    quote=shlex.quote
    base=f'{quote(python)} {quote(str(launcher))} --candidate {quote(str(source/"kernel.py"))} --artifacts ../verification-artifacts'
    return {'id':project_id or workload,'source':str(source),'actor_seconds':28800,'phase':'phase2',
      'goal':f'Implement {workload} in {language}. Export run(*args, **attrs) with the frozen PyTorch input order and all output tensors/states. Outputs must remain on the NPU, preserve inputs and their declared dtypes. Read the bundled pack specification; the frozen verifier covers every case. You may inspect prior projects and choose a starting source, but implement this project in its own source directory.',
      'performance_goal':'Pass every frozen numerical case and achieve geometric-mean speedup greater than 1 over the freshly qualified PyTorch baseline under paired measurements.',
      'verifier':{'files':[str(launcher),str(archive)],'command':base+' --mode correctness',
                  'performance_command':base+' --mode performance'}}


_LAUNCHER='''"""Trusted standalone launcher; only this file and its archive need transport."""
import fcntl,hashlib,json,os,sys,tarfile,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
archive=Path(__file__).with_name('verifier.tar.gz')
if hashlib.sha256(archive.read_bytes()).hexdigest()!='ARCHIVE_HASH':
    raise ValueError('Frozen verifier archive changed')
cache=Path(tempfile.gettempdir())/'rsi-baseline-verifiers';cache.mkdir(exist_ok=True,mode=0o700)
root=cache/'ARCHIVE_HASH'
with (cache/'ARCHIVE_HASH.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        intact=root.is_dir() and all((root/m.name).is_file() and hashlib.sha256((root/m.name).read_bytes()).digest()==hashlib.sha256(tar.extractfile(m).read()).digest() for m in members)
        if not intact:
            import shutil
            if root.exists():shutil.rmtree(root)
            root.mkdir();tar.extractall(root,filter='data')
    sys.path.insert(0,str(root))
from ascend_tools.baselines.project_verifier import main
raise SystemExit(main(root/'contract.json'))
'''
