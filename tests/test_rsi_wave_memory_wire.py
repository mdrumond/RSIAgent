"""Compact service admission preserves full immutable Memory and legacy waves."""
import hashlib
import json

import pytest

from benchmarks.ascend.subject_memory import SubjectMemory
from rsi.configuration import set_configuration, show_configuration
from rsi.service_wave import WaveBindings, memory_wire
from test_rsi_service_wave import wave_instance
from test_rsi_services import instance


@pytest.mark.parametrize('legacy', [False, True])
def test_wave_wire_uses_full_memory_in_real_runners_and_reopens_unchanged(
        wave_instance, tmp_path, monkeypatch, legacy):
    raw = wave_instance.store.put_bytes(b'complete origin instrumentation\n')
    memory = SubjectMemory(tmp_path/'accepted-memory', b'full initial context\n' * 8192)
    raw_refs = [raw] * 2048
    memory.accept('retained-record',
        dict(base_versions=[], kind='correctness', summary='Retain verified context'),
        dict(raw_refs=raw_refs))
    full = memory.snapshot()
    path = wave_instance.root/'memory/accepted.json'
    path.write_bytes(full)
    config = show_configuration(wave_instance.root)['payload']
    config.update(memory_snapshot=str(path), memory_sha256=hashlib.sha256(full).hexdigest())
    set_configuration(wave_instance.root, config)
    if legacy:
        prepare = WaveBindings._prepare
        def legacy_admission(self, maximum):
            wave = prepare(self, maximum)
            admitted = self.cycle.store.read_json(wave['admission_ref'])
            if 'memory_wire' in admitted:
                admitted.pop('memory_wire')
                admitted['memory_bytes'] = self.cycle.store.read_bytes(wave['memory']).hex()
                wave['admission_ref'] = self.cycle.store.put_json(admitted)
                self.cycle.scheduler.state['active_wave'] = wave
                self.cycle.scheduler._save()
            return wave
        monkeypatch.setattr(WaveBindings, '_prepare', legacy_admission)
    cycle = wave_instance.create_cycle('memory-wire', 'phase1')
    try:
        assert cycle.step().outcome == 'accepted'
        wave = cycle.scheduler.state['active_wave']
        admitted = cycle.store.read_json(wave['admission_ref'])
        assert cycle.store.read_bytes(admitted['memory']) == full
        wire = memory_wire(admitted)
        if legacy:
            assert wire == full
        else:
            assert len(wire) <= 256
            assert json.loads(wire)['reference'] == admitted['memory']
            assert 'memory_bytes' not in admitted
        pool = cycle.waves._pool(wave)
        assert bytes.fromhex(pool.plan['memory']) == wire
        rows = pool.status()['projects']
        pids = {key: value['worker']['pid'] for key, value in rows.items()}
        for op in wave['operations']:
            runner = cycle.open_project(op['id'])
            binding = json.loads((runner.root/'service-binding.json').read_text())
            assert binding['memory'] == admitted['memory']
            assert runner.store.read_bytes(binding['memory']) == full
            assert runner.store.read_json(binding['memory'])['memories'][0]['provenance']['raw_refs'] == raw_refs
        plan = pool.path.read_bytes()
    finally:
        cycle.close()
    reopened = wave_instance.open_cycle('memory-wire')
    try:
        retained = reopened.waves._pool(reopened.scheduler.state['active_wave'])
        retained.run(None)  # Completed children must never redispatch on resume.
        assert retained.path.read_bytes() == plan
        assert {key: row['worker']['pid'] for key, row in retained.status()['projects'].items()} == pids
    finally:
        reopened.close()


def test_compact_memory_wire_rejects_a_different_snapshot_reference():
    admitted = dict(memory='a' * 64, memory_wire=dict(schema='rsi-wave-memory-cas-v1', reference='b' * 64))
    with pytest.raises(ValueError):
        memory_wire(admitted)
