from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.phase1_adaptive_actor import AdaptiveActionContext
from benchmarks.a5kernels.knowledge import KnowledgeDB
from benchmarks.a5kernels.knowledge_agent import KnowledgeQuery
from benchmarks.ascend.phase1_knowledge import NativeAdaptiveKnowledge, adaptive_knowledge_identity
from benchmarks.ascend.production import NativeProductionConfig, build_native_adaptive_phase1
from test_memory_kdb_snapshot import Embeddings, source
from test_ascend_target_production import spec, NativeTransport


def config(tmp_path):
    text = tmp_path / 'guide.cpp'
    text.write_text('pipeline copy kernel\n')
    path = tmp_path / 'references.sqlite3'
    with KnowledgeDB(path, Embeddings()) as database:
        database.index(tmp_path, [text], collection='refs', language='ascend-c')
    return NativeProductionConfig(tmp_path / 'campaign', 'Ascend910B4', 'dav-c220', 'local',
        hashlib.sha256(b'Native guide').hexdigest(), 'a'*40, kdb=path, collection='refs',
        adaptive_kdb=True, curriculum_kdb=True)


def completion(calls):
    def complete(messages, seconds):
        calls.append((messages, seconds))
        packet = json.loads(messages[-1]['content'])
        candidate = packet['observation']['candidates'][0]
        return dict(text=json.dumps(dict(action='finish', selected=[dict(
            id=candidate['id'], reason='Applicable source excerpt')])), tokens=1)
    return complete


def actor_context(tmp_path):
    return AdaptiveActionContext(request=SimpleNamespace(branch_root=tmp_path/'branch', spec=spec()),
        prepared=SimpleNamespace(instruction='Find a bounded native copy example'), source=None, latest_execution=None,
        artifacts=ContentAddressedArtifacts(tmp_path/'artifacts'), remaining_seconds=30)


def test_production_actor_uses_compact_admitted_evidence_and_persistent_shared_budget(tmp_path):
    cfg, calls = config(tmp_path), []
    knowledge = NativeAdaptiveKnowledge(cfg, Embeddings(), completion(calls))
    context = actor_context(tmp_path)
    result = knowledge.actor_query(dict(query='pipeline', limit=1), context)
    assert result.observation['status'] == 'complete' and result.observation['result_count'] == 1
    assert len(json.dumps(result.observation).encode()) < 4096
    assert result.citations[0].collection == 'refs'
    for digest in result.admitted_artifacts:
        assert context.artifacts.read_bytes(digest)
    index = json.loads(context.artifacts.read_bytes(result.observation['results_index_sha256']))
    assert context.artifacts.read_bytes(index['results'][0]['artifact_sha256']) == b'pipeline copy kernel\n'
    assert calls[0][1] <= 30
    path = context.request.branch_root/'librarian-budget.json'
    state = json.loads(path.read_text())
    assert state['active_since'] is None and state['elapsed_seconds'] > 0
    state['elapsed_seconds'] = 90
    path.write_text(json.dumps(state))
    resumed = NativeAdaptiveKnowledge(cfg, Embeddings(), completion(calls))
    exhausted = resumed.actor_query(dict(query='pipeline', limit=1), context)
    assert exhausted.observation['status'] == 'insufficient' and not exhausted.citations
    assert len(calls) == 1


def test_curriculum_selection_handshake_and_actor_composition(tmp_path):
    from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
    from benchmarks.a5kernels.phase1_provider import direct_profile, DirectCompletion
    from test_a5_phase1_curriculum_selector import context, project, wave
    calls, decisions = [], []
    def transport(**request):
        decisions.append(request)
        text = '{"knowledge_query":{"query":"pipeline","limit":1}}' if len(decisions) == 1 else wave([project('selected')])
        return DirectCompletion(text, 1)
    controller = build_native_adaptive_phase1(config=config(tmp_path), native_transport=NativeTransport(),
        profile=direct_profile(MODEL_IDENTITIES[0]), guide_bytes=b'Native guide', objective='Learn native copy',
        initial_memory=b'', environment={'OPENAI_API_KEY':'fixture', 'DEEPSEEK_API_KEY':'fixture'},
        provider_transport=transport, embeddings=Embeddings(), librarian_complete=completion(calls))
    decision, session = controller.curriculum.select(context())
    assert decision.wave.projects[0].spec.project_id == 'selected'
    assert len(calls) == 1 and session.turns == 2
    packet = json.loads(calls[0][0][1]['content'])
    assert packet['role'] == 'curriculum' and packet['context']['target'] == 'Ascend910B4'
    assert list((tmp_path/'campaign'/'curriculum-librarian').glob('*/librarian-budget.json'))
    result = controller.execute_project.query_knowledge({'query':'pipeline', 'limit':1}, actor_context(tmp_path))
    assert result.citations and len(calls) == 2
    identity = json.loads((tmp_path/'campaign'/'campaign-identity.json').read_text())
    assert identity['adaptive_kdb']['model']['backend_model'] == 'deepseek-flash'


def test_fresh_memory_store_is_searchable_and_linked_artifact_read_is_gated(tmp_path):
    from benchmarks.a5kernels.memory_snapshot import export_snapshot
    cfg = config(tmp_path)
    original, digest = source(tmp_path/'producer')
    snapshot = export_snapshot([original], tmp_path/'snapshot')
    memory_db = tmp_path/'memory.sqlite3'
    with KnowledgeDB(memory_db, Embeddings()) as database:
        snapshot.index(database, collection='learned')
    cfg = replace(cfg, memory_kdb=memory_db, memory_collection='learned', memory_snapshot=snapshot.root)
    requests = []
    def complete(messages, seconds):
        requests.append(json.loads(messages[-1]['content']))
        if len(requests) == 1:
            value = dict(action='artifact', artifact=dict(collection='learned', digest=digest))
        else:
            packet = requests[-1]['observation']
            assert packet['artifact']['data'].startswith('pipeline code example')
            candidate = next(row for row in packet['candidates'] if row['citation']['collection'] == 'learned')
            value = dict(action='finish', selected=[dict(id=candidate['id'], reason='Read retained evidence')])
        return dict(text=json.dumps(value), tokens=1)
    knowledge = NativeAdaptiveKnowledge(cfg, Embeddings(), complete)
    assert knowledge.identity['snapshot'] == snapshot.fingerprint
    result = knowledge.actor_query({'query':'pipeline', 'limit':1}, actor_context(tmp_path))
    assert result.citations[0].collection == 'learned' and len(requests) == 2
    changed = replace(cfg, memory_collection='refs')
    with pytest.raises(ValueError, match='collides'):
        adaptive_knowledge_identity(changed)


def test_expired_actor_caller_and_changed_store_do_not_call_librarian(tmp_path):
    cfg, calls = config(tmp_path), []
    knowledge = NativeAdaptiveKnowledge(cfg, Embeddings(), completion(calls))
    context = actor_context(tmp_path)
    context = replace(context, remaining_seconds=0)
    with pytest.raises(TimeoutError):
        knowledge.actor_query({'query':'pipeline'}, context)
    assert not calls
    context = replace(context, remaining_seconds=30)
    path = tmp_path/'different.cpp'
    path.write_text('different content')
    cfg.kdb.unlink()
    with KnowledgeDB(cfg.kdb, Embeddings()) as database:
        database.index(tmp_path, [path], collection='refs', language='ascend-c')
    with pytest.raises(ValueError, match='identity changed'):
        knowledge.actor_query({'query':'pipeline'}, context)
    assert not calls


@pytest.mark.parametrize('fields', [dict(adaptive_kdb=True), dict(memory_kdb='only-one'),
    dict(librarian_seconds=0), dict(librarian_seconds=float('inf'))])
def test_configuration_rejects_unusable_adaptive_bindings(tmp_path, fields):
    with pytest.raises(ValueError):
        NativeProductionConfig(tmp_path, 'A3', 'arch', 'local', 'a'*64, 'b'*40, **fields)


def test_librarian_provider_is_deepseek_and_cancels_at_total_elapsed_deadline(monkeypatch):
    import asyncio
    import openai
    from benchmarks.ascend.phase1_knowledge import deepseek_librarian
    observed = []
    class Client:
        def __init__(self, **options):
            observed.append(options)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            observed.append('closed')
        async def create(self, **request):
            observed.append(request)
            try:
                await asyncio.sleep(10)
            finally:
                observed.append('cancelled')
    monkeypatch.setattr(openai, 'AsyncOpenAI', Client)
    complete = deepseek_librarian({'DEEPSEEK_API_KEY':'fixture'})
    with pytest.raises(TimeoutError):
        complete([{'role':'user', 'content':'JSON please'}], .01)
    assert observed[0]['max_retries'] == 0 and observed[0]['timeout'] == .01
    assert observed[1]['model'] == 'deepseek-flash'
    assert observed[-2:] == ['cancelled', 'closed']
