"""Distribution inventories remain immutable and independent of install locations."""
import base64
import csv
import hashlib
from importlib import metadata
from pathlib import Path
import zipfile

import pytest

from rsi import instances, instance_setup


def installed(root, *, location):
    contents = {'rsi/instances.py': b'owned release module\n',
        'integrations/pi/native-session.mjs': b'export const installed = true;\n',
        'rsi-0.1.0.dist-info/METADATA': b'Metadata-Version: 2.1\nName: rsi\nVersion: 0.1.0\n',
        'rsi-0.1.0.dist-info/direct_url.json': location.encode(),
        'rsi-0.1.0.dist-info/INSTALLER': location.encode()}
    for name, data in contents.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    record = root/'rsi-0.1.0.dist-info/RECORD'
    with record.open('w', newline='') as stream:
        writer = csv.writer(stream)
        for name, data in contents.items():
            sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            writer.writerow([name, 'sha256='+sha, len(data)])
        writer.writerow(['../../../bin/rsi', '', ''])
        writer.writerow(['rsi-0.1.0.dist-info/RECORD', '', ''])
    (root/'unowned-package').mkdir()
    (root/'unowned-package/private.txt').write_text('unrelated installed dependency')
    return metadata.PathDistribution(record.parent)


def test_owned_inventory_has_stable_identity_without_installer_paths(tmp_path, monkeypatch):
    receipts = []
    for name in ('first', 'second'):
        root = tmp_path/name
        distribution = installed(root, location=str(root))
        monkeypatch.setattr(instances, '__file__', str(root/'rsi/instances.py'))
        monkeypatch.setattr(instances.metadata, 'distribution', lambda _: distribution)
        destination = tmp_path/(name+'-snapshot')
        receipts.append(instances._copy_machinery(destination))
        assert (destination/'integrations/pi/native-session.mjs').read_bytes() == b'export const installed = true;\n'
        assert not (destination/'unowned-package').exists()
        assert not (destination/'rsi-0.1.0.dist-info/direct_url.json').exists()
        assert not (destination/'rsi-0.1.0.dist-info/INSTALLER').exists()
        rows = list(csv.reader((destination/'rsi-0.1.0.dist-info/RECORD').open()))
        assert all((destination/name).is_file() for name, _, _ in rows)
        assert {row[0] for row in rows} == {'rsi/instances.py', 'integrations/pi/native-session.mjs',
            'rsi-0.1.0.dist-info/METADATA', 'rsi-0.1.0.dist-info/RECORD'}
    assert receipts[0] == receipts[1]


@pytest.mark.parametrize('change', ['missing', 'changed', 'different-install'])
def test_snapshot_rejects_missing_changed_or_mismatched_installed_release(tmp_path, monkeypatch, change):
    root = tmp_path/'installed'
    distribution = installed(root, location='owned')
    monkeypatch.setattr(instances, '__file__', str(root/'rsi/instances.py'))
    if change == 'different-install':
        distribution = installed(tmp_path/'other', location='other')
    elif change == 'missing':
        (root/'integrations/pi/native-session.mjs').unlink()
    else:
        (root/'integrations/pi/native-session.mjs').write_text('modified after installation')
    monkeypatch.setattr(instances.metadata, 'distribution', lambda _: distribution)
    with pytest.raises(ValueError):
        instances._copy_machinery(tmp_path/'snapshot')


def test_reconstituted_wheel_excludes_resources_added_by_sdk_setup(tmp_path):
    source = tmp_path/'machinery'
    owned = {'rsi/__init__.py': b'value = 42\n',
        'rsi-0.1.0.dist-info/METADATA': b'Name: rsi\nVersion: 0.1.0\n',
        'rsi-0.1.0.dist-info/WHEEL': b'Wheel-Version: 1.0\nTag: py3-none-any\n',
        'rsi-0.1.0.dist-info/RECORD': b''}
    for name, data in owned.items():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    cached = source/'integrations/pi/node_modules/vendor/local.txt'
    cached.parent.mkdir(parents=True)
    cached.write_text('SDK cache is not an RSI wheel member')
    wheel = instance_setup._wheel(source, tmp_path, inventory=owned)
    with zipfile.ZipFile(wheel) as archive:
        assert set(archive.namelist()) == set(owned)
        assert archive.read('rsi/__init__.py') == owned['rsi/__init__.py']
