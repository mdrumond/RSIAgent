import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import os
import shutil
import subprocess
from ascend_tools import baselines
from ascend_tools.baselines.projects import prepare_project, _LAUNCHER
from ascend_tools.campaign_launch import inspect_launch


def launch_fixture(directory, target):
    identity = dict(target=target, transportTarget='private-' + target, runtimeName='named-runtime')
    identity['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    request = dict(target=target, profile={'environment': {'options': {
        'target': identity['transportTarget'], 'runtime': identity['runtimeName']}}},
        phase2Projects={}, qualification={'receipts': []})
    for workload in baselines.WORKLOADS:
        expected = baselines.describe(workload)
        receipt = dict(format=1, qualified=True, workload=workload, target=target,
            **{key: expected[key] for key in ('sourceHash', 'caseHash', 'verifierHash')},
            runtime=identity, independentChecks={'passed': True, 'workload': workload,
            'checkerHash': expected['independentCheckHash']}, cases=[dict(id=i, passed=True,
            referenceHash='r', outputHash='o', maxAbsError=0) for i in expected['caseIds']],
            timing={'passed': True, 'samples': [dict(id=i, seconds=[.01]) for i in expected['caseIds']]})
        request['qualification']['receipts'].append(receipt)
        request['phase2Projects'][workload] = prepare_project(workload, target,
            Path(directory) / target / workload, receipt, identity)
    return request


class LaunchTests(unittest.TestCase):
    def test_default_inspection_from_external_caller_directory(self):
        node = os.environ.get('TEST_NODE') or shutil.which('node')
        checkout = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            machinery = root / 'machinery'
            shutil.copytree(checkout / 'ascend_tools', machinery / 'ascend_tools',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            git = lambda *args: subprocess.check_output(['git', '-C', str(machinery), *args], text=True).strip()
            git('init', '-q'); git('add', '.')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'Frozen fixture')
            request = launch_fixture(root / 'projects', 'a3')
            request['profile'].update(machinery=str(machinery), python=os.sys.executable)
            request['profile']['environment']['module'] = 'machinery/integrations/environments/ascend.mjs'
            request['qualification']['localValidation'] = {'sourceRevision': git('rev-parse', 'HEAD')}
            for project in request['phase2Projects'].values():
                project['source'] = os.path.relpath(project['source'], root)
                project['verifier']['files'] = [os.path.relpath(p, root) for p in project['verifier']['files']]
            original = copy.deepcopy(request)
            script = "import {inspectCampaignLaunch} from %s;let data='';for await(const part of process.stdin)data+=part;const input=JSON.parse(data);await inspectCampaignLaunch(input);console.log(JSON.stringify(input));" % json.dumps((checkout / 'cli/campaign-manifest.mjs').as_uri())
            result = subprocess.run([node, '--max-old-space-size=768', '--input-type=module', '-e', script],
                input=json.dumps(request), text=True, cwd=root, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), original)

    def test_actual_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for target in ('a3', 'a5'):
                request = launch_fixture(directory, target)
                self.assertEqual(inspect_launch(request)['verified'], 7)
                project = request['phase2Projects']['fa']
                # Editable starter changes intentionally do not invalidate reference qualification.
                (Path(project['source']) / 'kernel.py').write_text('def run(): pass\n')
                self.assertEqual(inspect_launch(request)['verified'], 7)
                for field in ('target', 'runtime'):
                    changed = copy.deepcopy(request)
                    changed['profile']['environment']['options'][field] = 'different'
                    with self.assertRaises(ValueError): inspect_launch(changed)
                changed = copy.deepcopy(request)
                changed['phase2Projects']['fa'] = request['phase2Projects']['gdn']
                with self.assertRaises(ValueError): inspect_launch(changed)
                changed = copy.deepcopy(request)
                changed['phase2Projects']['fa']['verifier']['command'] = 'true'
                with self.assertRaises(ValueError): inspect_launch(changed)
                archive = Path(project['verifier']['files'][1])
                with tarfile.open(archive) as tar:
                    files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
                files['contract.json'] = b'{}'
                with tarfile.open(archive, 'w:gz') as tar:
                    for name, data in files.items():
                        member = tarfile.TarInfo(name); member.size = len(data)
                        tar.addfile(member, io.BytesIO(data))
                # Even a self-consistent replacement launcher cannot authorize changed contract bytes.
                Path(project['verifier']['files'][0]).write_text(_LAUNCHER.replace('ARCHIVE_HASH', baselines.file_hash(archive)))
                with self.assertRaises(ValueError): inspect_launch(request)


if __name__ == '__main__':
    unittest.main()
