"""Actual wheel imports/bootstrap and SDK tools, without checkout PYTHONPATH."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def run(argv, **kwargs):
    return subprocess.run(list(map(str, argv)), check=True, capture_output=True, text=True,
                          timeout=120, **kwargs)


@pytest.fixture(scope='module')
def installed_native(tmp_path_factory):
    work = tmp_path_factory.mktemp('installed-native')
    root = Path(__file__).resolve().parents[1]
    source = work/'source'
    if (root/'.git').exists():
        archive = work/'source.tar'
        run(['git', '-C', root, 'archive', '--format=tar', '--output', archive, 'HEAD'])
        source.mkdir()
        with tarfile.open(archive) as tree:
            for member in tree:
                if member.isfile():
                    path = source/member.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(tree.extractfile(member).read())
    else:
        shutil.copytree(root, source, ignore=shutil.ignore_patterns(
            '.git', '__pycache__', 'node_modules', '.pytest_cache', 'build', '*.egg-info'))
    wheels = work/'wheels'
    run([sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
         '--wheel-dir', wheels, source])
    environment = work/'venv'
    run([sys.executable, '-m', 'venv', '--without-pip', environment])
    python = environment/'bin/python'
    purelib = Path(run([python, '-I', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))']).stdout.strip())
    # Reuse installed declared dependencies only; RSI itself must come from this wheel.
    dependencies = [path for path in sys.path if path.endswith(('site-packages', 'dist-packages'))]
    (purelib/'test-dependencies.pth').write_text('\n'.join(dependencies)+'\n')
    clean = {key: value for key, value in os.environ.items() if not key.startswith('PYTHON')}
    run([python, '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', '--ignore-installed',
         next(wheels.glob('rsi-*.whl'))], cwd=work, env=clean)
    (purelib/'unrelated-installed-package.txt').write_text('must not enter instance machinery')
    def invoke(script, *args, env=None):
        result = run([python, '-I', '-c', script, *args], cwd=work, env={**clean, **(env or {})})
        return json.loads(result.stdout)
    return work, python, environment, invoke


def test_installed_distribution_bootstrap_has_private_data_and_dashboard(installed_native):
    work, python, environment, invoke = installed_native
    result = invoke('''
import json, sqlite3, sys
from importlib import metadata
from pathlib import Path
import rsi, coding.pi.native as native
from rsi.cli import main
from rsi.instances import create_instance
from benchmarks.ascend.subject_memory import SubjectMemory
from dashboard.app import create_app
root=Path(sys.argv[1])
assert Path(rsi.__file__).resolve().is_relative_to(Path(sys.prefix))
assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix))
assets=Path(native.__file__).resolve().parents[2]/'integrations/pi'
assert all((assets/name).is_file() for name in ('native-session.mjs','native-files.mjs'))
package=metadata.distribution('rsi')
requirements=package.metadata.get_all('Requires-Dist') or []
assert any(row.startswith('Flask==3.0.3') and 'extra ==' not in row for row in requirements)
assert 'roles' not in (package.metadata.get_all('Provides-Extra') or [])
kdb=root/'baseline.sqlite3'
with sqlite3.connect(kdb) as db:
    db.execute('create table facts(value text)');db.execute("insert into facts values('baseline')")
skills=root/'skills';skills.mkdir();(skills/'SKILL.md').write_text('Frozen reference skill')
mapping=root/'map.json';mapping.write_text(json.dumps({'design':['SKILL.md']}))
first,second=root/'first',root/'second'
a=create_instance(first,baseline_kdb=kdb,baseline_skills=skills,skill_map=mapping)
b=create_instance(second,baseline_kdb=kdb,baseline_skills=skills,skill_map=mapping)
assert a['machinery']==b['machinery'] and a['machinery']['kind']=='distribution'
assert not any('unrelated-installed-package' in name for name in a['files'])
with sqlite3.connect(first/'memory/baseline.sqlite3') as db:db.execute("insert into facts values('first-only')")
with sqlite3.connect(second/'memory/baseline.sqlite3') as db:assert db.execute('select value from facts').fetchall()==[('baseline',)]
one=SubjectMemory(first/'memory/accepted',b'first fixture')
two=SubjectMemory(second/'memory/accepted',b'second fixture')
assert json.loads(one.snapshot())['initial_memory']!=json.loads(two.snapshot())['initial_memory']
config=root/'dashboard.json';config.write_text(json.dumps({'campaigns':[],'rsi_instance':str(first),'controls':{'enabled':True}}))
app=create_app(config,root/'dashboard-cache.sqlite3')
response=app.test_client().get('/api/rsi/config')
assert response.status_code==200, response.get_data(as_text=True)
assert response.get_json()['payload']['kdb']=='memory/baseline.sqlite3'
print(json.dumps({'module':rsi.__file__,'machinery':a['machinery'],'dashboard_status':response.status_code,'assets':True}))
''', work, env={'PATH': '/no-git-required-for-installed-bootstrap'})
    assert result['assets'] and result['dashboard_status'] == 200
    assert Path(result['module']).is_relative_to(environment)
    assert len(result['machinery']['sha256']) == 64
    inspected = run([python, '-I', '-m', 'rsi', 'instance', 'inspect', work/'first'], cwd=work,
                    env={key: value for key, value in os.environ.items() if not key.startswith('PYTHON')})
    assert json.loads(inspected.stdout)['machinery'] == result['machinery']


def test_installed_cli_optional_repository_is_independent(installed_native):
    work, python, _, _ = installed_native
    inputs=work/'optional-inputs';inputs.mkdir()
    with sqlite3.connect(inputs/'baseline.sqlite3') as db:db.execute('create table facts(value text)')
    (inputs/'skills').mkdir();(inputs/'skills/SKILL.md').write_text('Frozen optional-repository fixture')
    (inputs/'map.json').write_text(json.dumps({'design':['SKILL.md']}))
    repo=work/'repository';repo.mkdir()
    for args in (['init','-q'],['config','user.name','Fixture'],['config','user.email','fixture@example.invalid']):
        run(['git','-C',repo,*args])
    (repo/'candidate.py').write_text('answer = 42\n')
    run(['git','-C',repo,'add','.']);run(['git','-C',repo,'commit','-qm','fixture'])
    revision=run(['git','-C',repo,'rev-parse','HEAD']).stdout.strip()
    clean={key:value for key,value in os.environ.items() if not key.startswith('PYTHON')}
    result=run([python,'-I','-m','rsi','instance','create',work/'with-repository',
        '--baseline-kdb',inputs/'baseline.sqlite3','--baseline-skills',inputs/'skills',
        '--skill-map',inputs/'map.json','--repository',repo],cwd=work,env=clean)
    assert json.loads(result.stdout)['repository']['revision']==revision
    clone=work/'with-repository/repository'
    assert run(['git','-C',clone,'rev-parse','HEAD']).stdout.strip()==revision
    assert run(['git','-C',clone,'remote']).stdout.strip()==''
    (clone/'candidate.py').write_text('independent edit\n')
    assert (repo/'candidate.py').read_text()=='answer = 42\n'


def test_actual_sdk_custom_tool_runs_from_installed_worker(installed_native):
    work, _, environment, invoke = installed_native
    from coding.pi.native import runtime_preflight
    node, modules = runtime_preflight()
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            delta=dict(role='assistant',content=None,tool_calls=[dict(index=0,id='installed-native-call',
                type='function',function=dict(name='verify_release',arguments='{}'))])
            chunk=dict(id='fixture',object='chat.completion.chunk',model='deepseek-flash',
                choices=[dict(index=0,delta=delta,finish_reason='tool_calls')],
                usage=dict(prompt_tokens=10,completion_tokens=2,total_tokens=12))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            self.wfile.write(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=invoke('''
import json,sys,time
from pathlib import Path
import coding.pi.native as native
profile=native.model_profile
native.model_profile=lambda name:dict(profile(name),base_url=sys.argv[2])
worker=Path(native.__file__).resolve().parents[2]/'integrations/pi/native-session.mjs'
assert worker.is_relative_to(Path(sys.prefix)) and worker.is_file()
calls=[]
def verify(args,context):
    calls.append({'operation':context['operation_id'],'call':context['tool_call_id']})
    return {'content':[{'type':'text','text':'Installed release verified'}],'terminate':True}
tool=native.NativeTool('verify_release','Verify the installed component.',{'type':'object','properties':{},'additionalProperties':False},verify)
with native.NativeSession(Path(sys.argv[1])/'installed-session',tools=[tool],environ={'DEEPSEEK_API_KEY':'fixture-secret'}) as session:
    deadline=time.monotonic()+30
    settled=session.prompt('Use verify_release.',operation_id='installed-proof',remaining=lambda:max(0.,deadline-time.monotonic()))
    assert settled.provider_settled and not settled.pending_tools
assert session.process.poll()==0
print(json.dumps({'worker':str(worker),'calls':calls,'settled':settled.provider_settled}))
''',work,f'http://127.0.0.1:{server.server_port}/v1', env={'RSI_PI_NODE':str(node), 'RSI_PI_NODE_MODULES':str(modules)})
    finally:
        server.shutdown();server.server_close();thread.join()
    assert len(requests)==1
    assert result['calls']==[{'operation':'installed-proof','call':'installed-native-call'}]
    assert result['settled'] and Path(result['worker']).is_relative_to(environment)
