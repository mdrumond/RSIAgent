import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from rsi.cli import main
from rsi.instances import create_instance, inspect_instance


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


@pytest.fixture
def inputs(tmp_path):
    kdb = tmp_path / 'baseline.db'
    with sqlite3.connect(kdb) as db:
        db.execute('create table facts (value text)')
        db.execute("insert into facts values ('known-good')")
    skills = tmp_path / 'skills'
    skills.mkdir()
    (skills / 'SKILL.md').write_text('baseline')
    (skills / '.env').write_text('secret')
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({'design': ['SKILL.md']}))
    return dict(baseline_kdb=kdb, baseline_skills=skills, skill_map=mapping)


def test_independent_instances(tmp_path, inputs):
    first, second = tmp_path / 'one', tmp_path / 'two'
    create_instance(first, **inputs)
    create_instance(second, **inputs)
    (first / 'skills/SKILL.md').write_text('changed')
    with sqlite3.connect(first / 'memory/baseline.db') as db:
        db.execute("insert into facts values ('new')")
    with sqlite3.connect(second / 'memory/baseline.db') as db:
        assert db.execute('select * from facts').fetchall() == [('known-good',)]
    assert (second / 'skills/SKILL.md').read_text() == 'baseline'
    assert inputs['baseline_skills'].joinpath('SKILL.md').read_text() == 'baseline'
    assert not (first / 'skills/.env').exists()
    for name in ('runs', 'artifacts', 'dashboard'):
        assert (first / name).is_dir()
    manifest = inspect_instance(first)
    assert all(not Path(name).is_absolute() and '..' not in Path(name).parts
               and '.git' not in Path(name).parts for name in manifest['files'])
    assert manifest['machinery']['revision']
    assert manifest['files']['machinery/rsi/cli.py']
    assert manifest['files']['machinery/rsi/instances.py']
    assert (first / 'machinery/rsi/instances.py').is_file()
    assert json.loads((first / 'config/instance.json').read_text())['dashboard']['port'] == 0


def test_existing_and_failed_creation_are_atomic(tmp_path, inputs, monkeypatch):
    target = tmp_path / 'instance'
    target.mkdir()
    with pytest.raises(FileExistsError):
        create_instance(target, **inputs)
    assert not list(target.iterdir())
    target.rmdir()
    def fail(*args, **kwargs):
        raise RuntimeError('copy failed')
    monkeypatch.setattr('rsi.instances._copy_machinery', fail)
    with pytest.raises(RuntimeError):
        create_instance(target, **inputs)
    assert not target.exists()
    assert not list(tmp_path.glob('.instance-*'))


def test_local_repository_pinned_independent_and_dirty_rejected(tmp_path, inputs):
    repo = tmp_path / 'source'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.email', 'test@example.com')
    git(repo, 'config', 'user.name', 'Test')
    (repo / 'main.py').write_text('print(1)')
    git(repo, 'add', 'main.py')
    git(repo, 'commit', '-qm', 'initial')
    revision = git(repo, 'rev-parse', 'HEAD')
    create_instance(tmp_path / 'instance', repository=repo, **inputs)
    clone = tmp_path / 'instance/repository'
    assert git(clone, 'rev-parse', 'HEAD') == revision
    assert not (clone / '.git/objects/info/alternates').exists()
    obj = next(p for p in (repo / '.git/objects').glob('*/*') if p.is_file())
    assert obj.stat().st_ino != (clone / '.git/objects' / obj.relative_to(repo / '.git/objects')).stat().st_ino
    assert git(repo, 'status', '--porcelain') == ''
    (repo / 'main.py').write_text('dirty')
    with pytest.raises(ValueError, match='clean'):
        create_instance(tmp_path / 'dirty-instance', repository=repo, **inputs)
    assert not (tmp_path / 'dirty-instance').exists()


def test_cli(tmp_path, inputs, capsys):
    args = ['instance', 'create', str(tmp_path / 'cli')]
    for key, value in inputs.items():
        args.extend(['--' + key.replace('_', '-'), str(value)])
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['schema_version'] == 1
    assert main(['instance', 'inspect', str(tmp_path / 'cli')]) == 0
    assert json.loads(capsys.readouterr().out)['files']


def test_sqlite_wal_is_included(tmp_path, inputs):
    db = sqlite3.connect(inputs['baseline_kdb'])
    try:
        db.execute('pragma journal_mode=wal')
        db.execute("insert into facts values ('wal')")
        db.commit()
        create_instance(tmp_path / 'wal-instance', **inputs)
        with sqlite3.connect(tmp_path / 'wal-instance/memory/baseline.db') as copied:
            assert copied.execute('select count(*) from facts').fetchone()[0] == 2
    finally:
        db.close()


def test_installed_release_inventory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rsi import instances
    installed = tmp_path / 'site-packages'
    (installed / 'rsi').mkdir(parents=True)
    (installed / 'rsi/instances.py').write_text('release')
    (installed / 'rsi/.env').write_text('secret')
    (installed / 'unlisted.txt').write_text('not release data')
    distribution = SimpleNamespace(
        version='0.1.0', files=[Path('rsi/instances.py'), Path('rsi/.env')],
        locate_file=lambda item: installed / item)
    monkeypatch.setattr(instances, '__file__', str(installed / 'rsi/instances.py'))
    monkeypatch.setattr(instances.metadata, 'distribution', lambda name: distribution)
    target = tmp_path / 'release'
    result = instances._copy_machinery(target)
    assert result == {'kind': 'distribution', 'revision': '0.1.0'}
    assert (target / 'rsi/instances.py').read_text() == 'release'
    assert not (target / 'rsi/.env').exists()
    assert not (target / 'unlisted.txt').exists()


def test_committed_release_only(tmp_path, monkeypatch):
    from rsi import instances
    repo = tmp_path / 'release-source'
    (repo / 'rsi').mkdir(parents=True)
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.email', 'test@example.com')
    git(repo, 'config', 'user.name', 'Test')
    (repo / 'rsi/instances.py').write_text('committed')
    (repo / '.env').write_text('secret')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'release')
    (repo / 'rsi/instances.py').write_text('uncommitted')
    (repo / 'untracked.txt').write_text('private')
    monkeypatch.setattr(instances, '__file__', str(repo / 'rsi/instances.py'))
    target = tmp_path / 'snapshot'
    instances._copy_machinery(target)
    assert (target / 'rsi/instances.py').read_text() == 'committed'
    assert not (target / '.env').exists()
    assert not (target / 'untracked.txt').exists()
    assert (repo / 'rsi/instances.py').read_text() == 'uncommitted'


def test_installed_wheel_bootstraps_outside_checkout(tmp_path, inputs):
    root = Path(__file__).resolve().parents[1]
    revision = git(root, 'rev-parse', 'HEAD')
    source = tmp_path / 'release-source'
    # Build input comes exclusively from pinned tracked Git blobs.
    entries = subprocess.check_output(
        ['git', '-C', str(root), 'ls-tree', '-rz', revision]).split(b'\0')
    for entry in filter(None, entries):
        info, name = entry.split(b'\t', 1)
        mode, kind, oid = info.split()
        if kind != b'blob' or mode == b'120000':
            continue
        target = source / os.fsdecode(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(
            ['git', '-C', str(root), 'cat-file', 'blob', oid.decode()]))
    subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
         '--wheel-dir', str(tmp_path / 'wheels'), str(source)],
        check=True, capture_output=True)
    installed = tmp_path / 'installed'
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile',
         '--target', str(installed), str(next((tmp_path / 'wheels').glob('rsi-*.whl')))],
        check=True, capture_output=True)
    environment = {**os.environ, 'PYTHONPATH': str(installed)}
    destination = tmp_path / 'instance'
    arguments = [sys.executable, '-m', 'rsi', 'instance', 'create', str(destination)]
    for key, value in inputs.items():
        arguments.extend(['--' + key.replace('_', '-'), str(value)])
    created = subprocess.run(arguments, cwd=tmp_path, env=environment,
                             check=True, capture_output=True, text=True)
    manifest = json.loads(created.stdout)
    assert manifest['machinery']['kind'] == 'distribution'
    assert 'machinery/rsi/instances.py' in manifest['files']
    assert all(not Path(name).is_absolute() and '.git' not in Path(name).parts
               for name in manifest['files'])
    environment['PYTHONPATH'] = str(destination / 'machinery')
    inspected = subprocess.run(
        [sys.executable, '-m', 'rsi', 'instance', 'inspect', str(destination)],
        cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    assert json.loads(inspected.stdout) == manifest
