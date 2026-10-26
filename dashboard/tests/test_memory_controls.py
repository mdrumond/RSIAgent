import json
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.memory_admission import MemoryAdmission
from benchmarks.ascend.project_verifier import ProjectReview
from benchmarks.ascend.subject_knowledge import SubjectKnowledge
from benchmarks.ascend.subject_memory import SubjectMemory, StaleMemory, digest
from benchmarks.ascend.subjects import Subject, SubjectScheduler
from dashboard.app import create_app
from dashboard.memory_controls import force_merge, reviewed_record


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def retained(tmp_path):
    store = ContentAddressedArtifacts(tmp_path / 'artifacts')
    source = store.put_bytes(b'float copy(float x) { return x; }\n')
    result = store.put_json(dict(host_verified=True, values=16))
    proposal = dict(operation='create', base_versions=[], summary='Vector copy preserves values',
        applicability='Vector movement', preconditions='Valid buffers', expectations='Equal outputs',
        limitations='One shape', before=None, before_unavailable_reason='First implementation',
        after=dict(sha256=source, start_line=1, end_line=1, text=store.read_bytes(source).decode()),
        evidence=[result], citations=[])
    review = dict(decision='accept', reason='Host values match the contract', evidence=[source, result],
                  repair=None, memory=proposal)
    request = ProjectReview('p', source, True, dict(source=source, result=result))
    record = dict(objective='Vector copies', wave_index=1, outcome=dict(host_verified=True,
        source_sha256=source, artifacts=dict(source=source, result=result)), review=review,
        review_sha256=store.put_json(review), review_request=asdict(request))
    write(tmp_path / 'subject-campaign.json', dict(waves=['wave-1'], projects={'p': record}, pending=None))
    scheduler = SubjectScheduler(tmp_path, [Subject('movement', 'Copies', 1)])
    scheduler.start_wave('wave-1', 'a' * 64)
    scheduler.complete_wave('wave-1', ['p'])
    scheduler.resolve_review('p', 'rejected')
    memory = SubjectMemory(tmp_path / 'memory', b'initial')
    knowledge = SubjectKnowledge(tmp_path, store, lambda: memory)
    services = SimpleNamespace(root=tmp_path, knowledge=knowledge)
    return services, record, proposal, scheduler


def test_manual_override_preserves_verification_and_original_verdict_and_replays(retained):
    services, record, proposal, scheduler = retained
    job = dict(review_id='p', status='rejected', evaluations=[dict(decision='reject', reason='No transfer gain')])
    write(services.root / 'memory-reviews' / digest('p') / 'job.json', job)
    receipt = force_merge(services, 'p', 'Useful for a future teaching example', 'manual-1')
    memory = services.knowledge.memory_getter()
    accepted = memory.records()[0]
    assert accepted['proposal'] == proposal
    assert accepted['provenance']['manual_override']['original_admission_status'] == 'rejected'
    assert accepted['provenance']['manual_override']['original_evaluations'] == job['evaluations']
    assert force_merge(services, 'p', 'Useful for a future teaching example', 'manual-1') == receipt
    assert len(memory.records()) == 1
    with pytest.raises(ValueError, match='different feedback'):
        force_merge(services, 'p', 'Changed reason', 'manual-1')
    with memory.transaction() as state:
        override = state['manual_overrides']['p']
    scheduler.override_review('p', override)
    assert scheduler.state()['reviews']['p']['original_status'] == 'rejected'
    assert scheduler.state()['reviews']['p']['status'] == 'accepted'
    assert record['outcome']['host_verified'] is True


def test_force_merge_refuses_failed_host_stale_bases_and_missing_evidence(retained):
    services, record, proposal, _ = retained
    root = services.root
    state = json.loads((root / 'subject-campaign.json').read_bytes())
    state['projects']['p']['outcome']['host_verified'] = False
    write(root / 'subject-campaign.json', state)
    with pytest.raises(ValueError, match='Host verification'):
        force_merge(services, 'p', 'Force it', 'one')
    state['projects']['p'] = record
    memory = services.knowledge.memory_getter()
    original = memory.accept('base', proposal, {})
    stale = dict(proposal, operation='append', base_versions=[original])
    memory.accept('replacement', dict(stale, summary='Updated base'), {})
    record['review']['memory'] = stale
    record['review_sha256'] = ContentAddressedArtifacts(root / 'artifacts').put_json(record['review'])
    write(root / 'subject-campaign.json', state)
    with pytest.raises(StaleMemory):
        force_merge(services, 'p', 'Force it', 'two')
    source = record['outcome']['source_sha256']
    store = ContentAddressedArtifacts(root / 'artifacts')
    (store.objects / source[:2] / source[2:]).unlink()
    with pytest.raises(OSError):
        reviewed_record(root, 'p')


def test_late_background_rejection_cannot_undo_manual_publication(retained):
    services, record, proposal, _ = retained
    started, release = threading.Event(), threading.Event()
    class Evaluation:
        def evaluate(self, *args, **options):
            started.set()
            assert release.wait(5)
            return dict(decision='reject', reason='No measured gain', evidence=proposal['evidence'], arms={})
    def forbidden(*args, **options):
        raise AssertionError('manual override must prevent new rewrite work')
    gate = MemoryAdmission(services.root / 'memory-reviews', services.knowledge.memory_getter(), Evaluation(), forbidden)
    try:
        gate.submit('p', proposal, {}, record['review_request'], remaining_seconds=30)
        assert started.wait(5)
        receipt = force_merge(services, 'p', 'Retain independently useful example', 'late')
        release.set()
        gate.wait(remaining_seconds=5)
        settled = gate.poll(remaining_seconds=5)[0]
        assert settled['outcome'] == 'accepted' and settled['manual_override']['admission'] == receipt
        job = json.loads(gate._path('p').read_bytes())
        assert job['evaluations'][0]['decision'] == 'reject' and job['status'] == 'accepted'
        assert services.knowledge.memory_getter().records()[0]['proposal'] == proposal
        from benchmarks.ascend.subject_campaign import SubjectCampaign
        campaign = SubjectCampaign(services.root, [Subject('movement', 'Copies', 1)], b'initial',
            select_wave=None, execute=None, verifier=None, complete=None, memory_gate=gate)
        campaign.state['memory_reviews'] = {'p': dict(outcome='rejected', subject_id='movement', sequence=1)}
        campaign.budget = SimpleNamespace(deadline=time.monotonic() + 30)
        campaign._settle_reviews()
        settled_review = campaign.state['memory_reviews']['p']
        assert settled_review['outcome'] == 'accepted' and settled_review['original_review']['outcome'] == 'rejected'
        assert campaign.scheduler.state()['reviews']['p']['status'] == 'accepted'
    finally:
        release.set()
        gate.close()


def test_memory_route_requires_csrf_and_publishes_with_real_memory_owner(retained, tmp_path, monkeypatch):
    from contextlib import contextmanager
    import dashboard.subject_services as binding
    services, _, _, _ = retained
    config = tmp_path / 'dashboard.json'
    write(config, dict(controls=dict(enabled=True), subject_campaigns=[dict(id='research', target='a3',
        root=str(services.root), config='configured-runtime', initial_memory='configured-seed')]))
    @contextmanager
    def bind(*args, **kwargs):
        yield services
    monkeypatch.setattr(binding, 'bind_subject', bind)
    app = create_app(config, tmp_path / 'cache.sqlite')
    client = app.test_client()
    url = '/api/campaigns/research/projects/p/memory/force'
    payload = dict(reason='Useful example', confirmation='force-memory:p')
    try:
        assert client.post(url, json=payload).status_code == 403
        token = client.get('/api/csrf').get_json()['token']
        response = client.post(url, json=payload, headers={'X-CSRF-Token': token, 'Idempotency-Key': 'force-route'})
        assert response.status_code == 202
        operation_id = response.get_json()['operation_id']
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            operation = client.get('/api/operations/' + operation_id).get_json()
            if operation['status'] not in {'queued', 'running'}:
                break
            time.sleep(.01)
        assert operation['status'] == 'succeeded', operation
        memory = services.knowledge.memory_getter()
        assert memory.read(operation['result']['admission']['version']) == memory.records()[0]
        assert services.knowledge.memory_getter().records()[0]['proposal']['summary'] == 'Vector copy preserves values'
    finally:
        app.extensions['dashboard_controls'].close()


def test_index_failure_never_publishes_manual_memory(retained):
    services, _, _, _ = retained
    memory = services.knowledge.memory_getter()
    before = memory.snapshot()
    def failed_index(*args):
        raise OSError('Knowledge database unavailable')
    services.knowledge.index = failed_index
    with pytest.raises(OSError):
        force_merge(services, 'p', 'Retain useful example', 'index-failure')
    assert memory.snapshot() == before
    with memory.transaction() as state:
        assert 'p' not in state.get('manual_overrides', {})
