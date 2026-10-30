"""Exercise the public CLI in a separate interpreter, without any provider calls."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rsi.configuration import show_configuration


@pytest.fixture
def instance(tmp_path):
    root = tmp_path / 'instance'
    (root / 'config').mkdir(parents=True)
    (root / 'instance.json').write_text(json.dumps({'schema_version': 1}))
    bootstrap = {'machinery': 'machinery', 'kdb': 'memory/base.db',
                 'skills': 'skills', 'skill_map': 'config/map.json',
                 'dashboard': {'host': '127.0.0.1', 'port': 0}}
    (root / 'config/instance.json').write_text(json.dumps(bootstrap))
    return root


def cli(instance, *arguments):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-m', 'rsi', '--instance', str(instance), *map(str, arguments)],
        cwd=instance.parent, env={**os.environ, 'PYTHONPATH': str(root)},
        capture_output=True, text=True)
    return result, json.loads(result.stdout) if result.returncode == 0 else None


def inventory(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def write_input(tmp_path, payload):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(payload))
    return path


def test_reads_are_nonmutating_and_bootstrap_compatible(instance):
    before = inventory(instance)
    result, first = cli(instance, 'config', 'show')
    assert result.returncode == 0, result.stderr
    assert first == show_configuration(instance)
    assert first['payload']['kdb'] == 'memory/base.db'
    assert first['payload']['provider']['profile_id'] == 'deepseek-flash-native-v1'
    result, folders = cli(instance, 'folders', 'list')
    assert result.returncode == 0, result.stderr
    assert folders['path'] == str(instance)
    assert inventory(instance) == before
    assert not (instance / 'config/revisions').exists()


def test_revision_history_and_detached_old_inputs(instance, tmp_path):
    old = show_configuration(instance)
    original = inventory(instance)
    payload = dict(old['payload'])
    payload.update({
        'phase1_projects': [{'project_id': 'micro', 'goal': 'sum',
                             'command': ['python', 'sum.py']}],
        'phase2_projects': [{'project_id': 'app', 'goal': 'build',
                             'command': ['python', 'app.py']}],
        'budgets': {'actor_seconds': None, 'execution_tries': None},
        'prompts': {'actor': 'Check numeric results', 'skillwriter': 'Prefer recovery'},
        'roles_python': '/opt/roles/python', 'roles': {'review': {'temperature': 0}},
        'tuning': {'every_projects': 7, 'advanced': {'selection': 'paired'}},
        'advanced': {'future': [1, 2]}})
    path = write_input(tmp_path, payload)
    result, new = cli(instance, 'config', 'set', '--file', path,
                      '--expect-revision', old['revision'])
    assert result.returncode == 0, result.stderr
    assert new['revision'] != old['revision']
    assert new['payload'] == payload
    assert json.loads((instance / f"config/revisions/{old['revision']}.json").read_text()) == old['payload']
    retained = inventory(instance)
    # A subsequent edit cannot mutate a captured task input or an earlier revision.
    path.write_text(json.dumps({'prompts': {'actor': 'New instructions'}}))
    result, latest = cli(instance, 'config', 'set', '--file', path,
                         '--expect-revision', new['revision'])
    assert result.returncode == 0, result.stderr
    assert latest['payload']['kdb'] == old['payload']['kdb']
    assert new['payload']['prompts']['actor'] == 'Check numeric results'
    for key, value in retained.items():
        if key != 'config/current.json':
            assert (instance / key).read_bytes() == value
    assert (instance / 'config/instance.json').read_bytes() == original['config/instance.json']
    assert show_configuration(instance) == latest


@pytest.mark.parametrize('payload', [
    {'phase1_projects': [{'project_id': 'x', 'goal': 'run', 'command': []}]},
    {'phase2_projects': [{'project_id': 'x', 'command': ['python']}]},
    {'budgets': {'actor_seconds': -1}},
    {'budgets': {'execution_tries': True}},
    {'budgets': {'actor_second': 3}},
    {'prompts': {'actor': 8}},
    {'browser_roots': []},
    {'provider': []},
])
def test_invalid_inputs_do_not_publish(instance, tmp_path, payload):
    before = inventory(instance)
    result, _ = cli(instance, 'config', 'set', '--file', write_input(tmp_path, payload))
    assert result.returncode == 2
    assert inventory(instance) == before
    assert not (instance / 'config/revisions').exists()


def test_expected_revision_conflict_has_no_side_effects(instance, tmp_path):
    path = write_input(tmp_path, {'prompts': {'actor': 'first'}})
    before = inventory(instance)
    result, _ = cli(instance, 'config', 'set', '--file', path, '--expect-revision', '0' * 64)
    assert result.returncode == 2
    assert inventory(instance) == before
    assert not (instance / 'config/revisions').exists()
    base = show_configuration(instance)['revision']
    result, first = cli(instance, 'config', 'set', '--file', path, '--expect-revision', base)
    assert result.returncode == 0, result.stderr
    before = inventory(instance)
    path.write_text(json.dumps({'prompts': {'actor': 'stale writer'}}))
    result, _ = cli(instance, 'config', 'set', '--file', path, '--expect-revision', base)
    assert result.returncode == 2
    assert inventory(instance) == before
    assert show_configuration(instance) == first


def test_folder_containment_symlinks_and_configured_root(instance, tmp_path):
    allowed = instance / 'projects'
    allowed.mkdir()
    (allowed / 'child').mkdir()
    (allowed / 'file.py').write_text('pass')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (allowed / 'escape').symlink_to(outside, target_is_directory=True)
    (allowed / 'alias').symlink_to(allowed / 'child', target_is_directory=True)
    path = write_input(tmp_path, {'browser_roots': ['projects']})
    result, _ = cli(instance, 'config', 'set', '--file', path)
    assert result.returncode == 0, result.stderr
    before = inventory(instance)
    result, folders = cli(instance, 'folders', 'list')
    assert result.returncode == 0, result.stderr
    assert folders['path'] == str(allowed)
    assert {item['name'] for item in folders['folders']} == {'alias', 'child'}
    for target in (outside, 'projects/escape', 'projects/../..', 'projects/file.py'):
        result, _ = cli(instance, 'folders', 'list', target)
        assert result.returncode == 2
    result, folders = cli(instance, 'folders', 'list', 'projects/alias')
    assert result.returncode == 0, result.stderr
    assert folders['path'] == str(allowed / 'child')
    assert inventory(instance) == before


def test_missing_instance_and_escaped_internal_state(tmp_path, instance):
    before = inventory(instance)
    for arguments in [('config', 'show'), ('folders', 'list')]:
        result, _ = cli(tmp_path / 'absent', *arguments)
        assert result.returncode == 2
    assert not (tmp_path / 'absent').exists()
    external = tmp_path / 'external.json'
    external.write_text('{}')
    (instance / 'config/current.json').symlink_to(external)
    result, _ = cli(instance, 'config', 'show')
    assert result.returncode == 2
    assert external.read_text() == '{}'
    assert (instance / 'config/instance.json').read_bytes() == before['config/instance.json']


def test_concurrent_cli_writers_cannot_lose_update(instance, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    revision = show_configuration(instance)['revision']
    inputs = []
    for index in range(2):
        path = tmp_path / f'writer-{index}.json'
        path.write_text(json.dumps({'prompts': {'actor': f'writer-{index}'}}))
        inputs.append(path)
    def publish(path):
        return cli(instance, 'config', 'set', '--file', path,
                   '--expect-revision', revision)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, inputs))
    assert sorted(result.returncode for result, _ in results) == [0, 2]
    winner = next(payload for result, payload in results if result.returncode == 0)
    assert show_configuration(instance) == winner
    assert len(list((instance / 'config/revisions').glob('*.json'))) == 2
