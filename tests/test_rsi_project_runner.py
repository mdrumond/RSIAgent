"""Production composition with scripted decisions and real CPU/KDB services."""
import sys

import pytest

from benchmarks.a5kernels.knowledge import KnowledgeDB
from rsi.contracts import BudgetLimits, ProjectSpec, Stage
from rsi.project_runner import ProjectRunner
from rsi.skills import builtin_bundle
from rsi.strategy import knowledge_db_query


class TestEmbedding:
    model, revision = 'TEST', '1'

    def embed(self, texts):
        return [[1., float('stable' in text)] for text in texts]


class Session:
    def __init__(self, *, missing_query=False, repair=True, redirect=False, perform_label=None):
        self.missing_query, self.repair = missing_query, repair
        self.perform_label = perform_label
        self.redirect = redirect
        self.calls = 0

    def prepare(self, timeout, *, schemas):
        self.schemas = schemas

    def next_action(self, context, allowed, **kwargs):
        self.calls += 1
        for name in context['skills']:
            if name not in [row['skill'] for row in context['read_skills']]:
                return 'read_skill', {'name': name}
        evidence = {row['name']: row['result'] for row in context['evidence']}
        stage = Stage(context['stage'])
        if stage in (Stage.DESIGN, Stage.OPTIMISATION) and not self.missing_query:
            if 'query_kdb' not in evidence:
                return 'query_kdb', {'query': 'stable sum'}
            if 'retrospect' not in evidence:
                return 'retrospect', {'supplement': 'Retain numeric failure; use stable accumulation.'}
            if 'select_strategy' not in evidence:
                response = evidence['query_kdb']['evidence']['payload']['response']
                decisions = [dict(group=g, index=i, action='used' if g == 'results' else 'rejected',
                    reason='Stable reference' if g == 'results' else 'Avoid observed numeric cancellation')
                    for g in ('results', 'negative_records') for i in range(len(response[g]))]
                if self.redirect:
                    for d in decisions:
                        if d['group'] == 'negative_records':
                            d.update(action='used', redirection='Replace naive accumulation with fsum')
                return 'select_strategy', dict(proposal='Use stable accumulation',
                    query_ref=evidence['query_kdb']['reference'],
                    retrospective_ref=evidence['retrospect']['reference'],
                    skill_refs=[r['receipt_id'] for r in context['read_skills']], decisions=decisions)
            if stage == Stage.OPTIMISATION and 'optimisation_decision' not in evidence:
                return 'optimisation_decision', dict(action='noop', reason='Correctness established; preserve stable sum')
        args = {'proposal': self.perform_label or 'Use stable accumulation'}
        if stage == Stage.DEBUG and self.repair and any(
                row['classification'] == 'numeric_failure' for row in context['task']['feedback']):
            snapshot = context['source']
            args['edit'] = dict(path=snapshot['path'], sha256=snapshot['sha256'],
                                source='from math import fsum\ndef solve(xs): return fsum(xs)\n')
        return 'perform', args

    def close(self):
        pass


def make(tmp_path, session=None, *, config_overrides=None, budgets=None, source_text=None, **overrides):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text(source_text or 'def solve(xs):\n    total = 0.0\n    for x in xs: total += x\n    return total\n')
    (source / 'helper.py').write_text('value = 1\n')
    verifier = tmp_path / 'verify.py'
    verifier.write_text("import runpy\nsolve = runpy.run_path('solution.py')['solve']\n"
                        "assert solve([1e16, 1., -1e16]) == 1., 'numeric cancellation'\n")
    database = KnowledgeDB(tmp_path / 'knowledge.sqlite3', TestEmbedding())
    database.index(source, [source / 'solution.py'], collection='examples', language='python', revision='abc')
    query = knowledge_db_query(database, ('examples',), negative_query=lambda _: [
        dict(record_id='unstable', reason='cancellation', provenance={'artifact': 'numeric-failure'})])
    injected = dict(query=query, backend_identity='TEST@1', test_backend=True,
                    session_factory=lambda: session or Session())
    injected.update(overrides)
    spec = ProjectSpec('sum', 'Stable numeric sum', (sys.executable, 'solution.py'),
                       budgets=budgets or BudgetLimits(execution_tries=40))
    config = dict(source=str(source), verifier=str(verifier),
        editable_files=['solution.py', 'helper.py'], commands={
            'compile': [sys.executable, '-m', 'py_compile', 'solution.py'],
            'test': [sys.executable, '{verifier}'],
            'validation': [sys.executable, '{verifier}']})
    config.update(config_overrides or {})
    runner = ProjectRunner.create(tmp_path / 'task', spec, builtin_bundle(), config, **injected)
    return runner, injected, database


@pytest.mark.parametrize('redirect', [False, True])
def test_nine_stages_numeric_repair_and_real_negative_evidence(tmp_path, redirect):
    runner, _, database = make(tmp_path, Session(redirect=redirect))
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        assert result['correctness'] is True
        assert set(Stage(row['evidence']['stage']) for row in result['history']) == set(Stage)
        assert any(not row['passed'] and row['evidence']['stage'] == 'test' for row in result['history'])
        selections = [runner.store.read_json(e['reference'])['payload'] for e in runner.events
                      if e['type'] == 'project-tool' and e['name'] == 'select_strategy']
        assert all(any(d['group'] == 'negative_records' and d['action'] == ('used' if redirect else 'rejected')
                       for d in s['decisions']) for s in selections)
        assert runner.workflow.accounting.seconds('compilation') > 0
        assert runner.workflow.accounting.seconds('thinking') > 0
        assert runner.backend.verifier.parent != runner.backend.project
        assert runner.backend.read('solution.py')['content'].startswith('from math import fsum')
    finally:
        database.close()


def test_missing_query_refuses_strategy_before_host_execution(tmp_path):
    runner, _, database = make(tmp_path, Session(missing_query=True))
    try:
        result = runner.run()
        assert result['status'] == 'suspended' and result['stage'] == 'design'
        assert not any(e['type'] == 'command-results' for e in runner.events)
        assert not runner.workflow.outcome
    finally:
        database.close()


def test_pending_resume_never_calls_provider_twice(tmp_path):
    session = Session(missing_query=True)
    runner, injections, database = make(tmp_path, session)
    try:
        first = runner.run()
        calls = session.calls
        reopened = ProjectRunner(runner.root, **injections)
        second = reopened.run()
        assert second['pending'] == first['pending']
        assert second['status'] == 'suspended'
        assert session.calls == calls
    finally:
        database.close()


def test_multiple_files_selection_and_stale_edit_refusal(tmp_path):
    runner, _, database = make(tmp_path)
    try:
        runner.actor.step(runner.context())
        runner.actor.step(runner.context())
        op = runner.workflow.begin()
        receipt = runner.tool('read_project_file', Stage.DESIGN, {'path': 'helper.py'}, op)
        snapshot = runner.snapshot()
        assert snapshot.path == 'helper.py'
        assert runner.store.read_json(receipt['reference'])['content'] == 'value = 1\n'
        (runner.backend.project / 'helper.py').write_text('value = 2\n')
        from rsi.runtime import WorkflowError
        with pytest.raises(WorkflowError, match='source changed'):
            runner.backend.edit('helper.py', 'value = 3\n', expected_file_hash=snapshot.sha256,
                                expected_project_hash=snapshot.project_hash)
    finally:
        runner.actor.close()
        database.close()


def test_unavailable_query_suspends_instead_of_empty_result(tmp_path):
    def unavailable(_):
        raise OSError('backend offline')
    runner, _, database = make(tmp_path, query=unavailable)
    try:
        result = runner.run()
        assert result['status'] == 'suspended' and result['stage'] == 'design'
        assert any(e.get('kind') == 'query-unavailable' for e in runner.events)
        assert not any(e.get('kind') == 'selection' for e in runner.events)
    finally:
        database.close()


def test_recorded_stage_result_reconciles_without_provider_replay(tmp_path):
    from rsi.runtime import WorkflowError
    session = Session()
    runner, injections, database = make(tmp_path, session)
    try:
        runner.actor.step(runner.context())
        def interrupted(*args, **kwargs):
            raise WorkflowError('interrupted before checkpoint')
        runner.workflow.complete = interrupted
        with pytest.raises(WorkflowError, match='interrupted'):
            runner.actor.step(runner.context())
        runner.actor.close()
        calls = session.calls
        reopened = ProjectRunner(runner.root, **injections)
        reopened.reconcile()
        assert reopened.workflow.stage == Stage.DESIGN
        assert reopened.workflow.pending is None
        assert session.calls == calls
    finally:
        database.close()


def test_budget_exhaustion_keeps_partial_history_without_acceptance(tmp_path):
    runner, _, database = make(tmp_path)
    try:
        runner.actor.step(runner.context())
        # Exhaust actual charged time without inventing a stage result.
        token = runner.workflow.accounting.start('thinking')
        runner.workflow.accounting.end(token)
        runner.workflow.limits = BudgetLimits(actor_seconds=1e-9)
        result = runner.run()
        assert result['status'] == 'budget_exhausted'
        assert len(result['history']) == 1
        assert runner.workflow.outcome is None
    finally:
        database.close()


def test_correctness_and_performance_shortfall_are_separate(tmp_path):
    runner, _, database = make(tmp_path,
        source_text='from math import fsum\ndef solve(xs): return fsum(xs)\n',
        budgets=BudgetLimits(design_iterations=2, implementation_iterations=1, execution_tries=40),
        config_overrides=dict(
        performance_seconds=1e-9,
        commands={
            'compile': [sys.executable, '-m', 'py_compile', 'solution.py'],
            'test': [sys.executable, '{verifier}'],
            'validation': [sys.executable, '{verifier}'],
            'timing': [sys.executable, 'solution.py']}))
    try:
        result = runner.run()
        assert result['status'] == 'budget_exhausted', result
        assert result['correctness'] is True
        assert result['performance'] == 'shortfall'
        assert runner.workflow.outcome is None
        validations = [r for r in result['history'] if r.get('evidence', {}).get('stage') == 'validation']
        assert len(validations) == 2 and all(not r['passed'] for r in validations)
        assert all(e['correctness'] is True for e in runner.events if e['type'] == 'validation-results')
        designs = [r for r in result['history'] if r.get('evidence', {}).get('stage') == 'design']
        assert len(designs) == 2
        first, second = [r['evidence'] for r in designs]
        assert first['retrieval_refs'] != second['retrieval_refs']
        assert first['retrospective_ref'] != second['retrospective_ref']
        retrospective = runner.store.read_json(second['retrospective_ref'])['payload']
        assert any(r.get('evidence', {}).get('stage') == 'validation' and not r['passed']
                   for r in retrospective['history'])
        (runner.backend.project / 'solution.py').write_text('invalid source!')
        reopened = runner.run()
        assert reopened['status'] == 'budget_exhausted'
        assert reopened['correctness'] is None
    finally:
        database.close()


def test_perform_label_can_paraphrase_verified_selection(tmp_path):
    runner, _, database = make(tmp_path, Session(perform_label='Apply the chosen stable approach'))
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        assert runner.store.read_json(result['contract_ref'])['spec']['goal'] == runner.spec.goal
        assert runner.store.read_json(result['execution_contract_ref']) == runner.backend.contract
    finally:
        database.close()


class Clock:
    value = 1000.

    def __call__(self):
        return self.value


def test_copy_consumes_wall_budget_but_not_actor_budget(tmp_path, monkeypatch):
    import rsi.python_execution as execution
    clock, session = Clock(), Session()
    copy = execution.shutil.copytree
    def delayed_copy(*args, **kwargs):
        result = copy(*args, **kwargs)
        clock.value += 20
        return result
    monkeypatch.setattr(execution.shutil, 'copytree', delayed_copy)
    runner, _, database = make(tmp_path, session, clock=clock,
        budgets=BudgetLimits(actor_seconds=30, wall_seconds=10))
    try:
        result = runner.run()
        assert result['status'] == 'budget_exhausted', result
        assert runner.workflow.state['created'] == 1000.
        assert runner.workflow.remaining() == dict(actor_seconds=30., wall_seconds=0.)
        assert result['history'] == [] and result['pending'] is None
        assert session.calls == 0
    finally:
        database.close()


@pytest.mark.parametrize('error_type,classification', [
    ('ActionError', 'malformed_action'), ('PiProviderError', 'provider_failure'),
    ('PiDeadlineError', 'provider_timeout'), ('PiCancelled', 'actor_cancelled'),
    ('PiError', 'actor_failure')])
def test_known_pi_failure_is_durable_charged_and_never_replayed(tmp_path, error_type, classification):
    from coding.pi import rpc, session as pi_session
    error = getattr(pi_session if error_type == 'ActionError' else rpc, error_type)
    clock = Clock()
    class Failed(Session):
        def prepare(self, timeout, *, schemas):
            clock.value += 100  # Actor setup is excluded.
        def next_action(self, *args, **kwargs):
            self.calls += 1
            clock.value += 2
            raise error('controlled provider/action failure')
    session = Failed()
    runner, injections, database = make(tmp_path, session, clock=clock)
    try:
        result = runner.run()
        assert result['status'] == 'suspended'
        assert result['classification'] == classification
        assert result['pending'] and runner.workflow.outcome is None
        assert runner.workflow.accounting.seconds('thinking') == 2
        assert runner.workflow.accounting.seconds('setup') == 100
        assert runner.latest('project-failure')['classification'] == classification
        reopened = ProjectRunner(runner.root, **injections).run()
        assert reopened['pending'] == result['pending']
        assert session.calls == 1
    finally:
        database.close()


def test_concurrent_event_append_preserves_every_cas_record(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import json
    runner, _, database = make(tmp_path)
    barrier = Barrier(8)
    def append(index):
        barrier.wait(timeout=5)
        runner.append(dict(type='provider-record', index=index))
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(append, range(8)))
        refs = json.loads(runner.event_path.read_text())
        assert refs == runner.event_refs
        assert [runner.store.read_json(ref) for ref in refs] == runner.events
        assert {e['index'] for e in runner.events} == set(range(8))
    finally:
        database.close()


def test_global_task_construction_queries_and_advertised_scopes(tmp_path):
    class GlobalReader(Session):
        def next_action(self, context, allowed, **kwargs):
            stage = Stage(context['stage'])
            assert ('select_strategy' in allowed) == (stage in (Stage.DESIGN, Stage.OPTIMISATION))
            assert ('optimisation_decision' in allowed) == (stage == Stage.OPTIMISATION)
            assert ('request_redesign' in allowed) == (stage in (Stage.DEBUG, Stage.VALIDATION))
            if 'perform' in allowed:
                assert set(context['skills']) <= {r['skill'] for r in context['read_skills']}
                if stage in (Stage.DESIGN, Stage.OPTIMISATION):
                    assert any(r['name'] == 'select_strategy' for r in context['evidence'])
                if stage == Stage.OPTIMISATION:
                    assert any(r['name'] == 'optimisation_decision' for r in context['evidence'])
            if stage == Stage.TASK_CONSTRUCTION:
                evidence = {r['name'] for r in context['evidence']}
                if 'query_kdb' not in evidence:
                    return 'query_kdb', dict(query='stable sum')
                if 'retrospect' not in evidence:
                    return 'retrospect', dict(supplement='Inspect starting task context')
            return super().next_action(context, allowed, **kwargs)
    runner, _, database = make(tmp_path, GlobalReader())
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        rows = [runner.store.read_json(e['reference']) for e in runner.events
                if e['type'] == 'strategy-evidence' and e['kind'] == 'retrieval']
        assert any(r['context']['stage'] == Stage.TASK_CONSTRUCTION.value for r in rows)
    finally:
        database.close()


def test_proven_admission_rejection_preserves_counts_time_and_source(tmp_path):
    from rsi.runtime import WorkflowError
    runner, injections, database = make(tmp_path)
    try:
        flow = runner.workflow
        flow.state['stage'] = Stage.DESIGN.value
        operation = flow.begin()
        flow.dispatched(operation, 'pi:closed-admission')
        token = flow.accounting.start('thinking')
        flow.accounting.end(token)
        counts, charged, source = (flow.state['design'], flow.state['executions']), flow.accounting.seconds(), flow.source_hash
        with pytest.raises(WorkflowError, match='untrusted'):
            flow.reject_operation(operation, 'unknown-proof', source)
        proof = dict(handle=flow.pending['handle'], actor_closed=True, no_host_effects=True,
                     observation='Host observed Pi session closed after rejected tool admission')
        reference = runner.record_admission_failure(operation, controller_proof=proof, observed_source_hash=source)
        with pytest.raises(WorkflowError, match='source mismatch'):
            flow.reject_operation(operation, reference, 'changed-source')
        reopened = ProjectRunner(runner.root, **injections)
        reopened.workflow.reject_operation(operation, reference, source)
        assert reopened.workflow.pending is None
        assert (reopened.workflow.state['design'], reopened.workflow.state['executions']) == counts
        assert reopened.workflow.accounting.seconds() == charged
        assert reopened.workflow.source_hash == source
        assert reopened.workflow.state['history'][-1]['host_failure_ref'] == reference
        assert reopened.workflow.begin() != operation
    finally:
        database.close()


def test_source_change_or_host_effects_veto_closure_proof(tmp_path):
    from rsi.runtime import WorkflowError
    runner, _, database = make(tmp_path)
    try:
        flow = runner.workflow
        operation = flow.begin()
        flow.dispatched(operation, 'pi:unknown')
        proof = dict(handle='pi:unknown', actor_closed=True, no_host_effects=True,
                     observation='Controller closure observation')
        reference = runner.record_admission_failure(operation, controller_proof=proof,
                                                     observed_source_hash=flow.source_hash)
        path = runner.backend.project / 'solution.py'
        original = path.read_text()
        path.write_text(original + '\n# external change\n')
        with pytest.raises(WorkflowError, match='untrusted'):
            flow.reject_operation(operation, reference, flow.source_hash)
        path.write_text(original)
        runner.append(dict(type='host-dispatch', operation_id=operation))
        with pytest.raises(WorkflowError, match='no host effects'):
            runner.record_admission_failure(operation, controller_proof=proof,
                                             observed_source_hash=flow.source_hash)
        with pytest.raises(WorkflowError, match='untrusted'):
            flow.reject_operation(operation, reference, flow.source_hash)
        assert flow.pending['id'] == operation
    finally:
        database.close()


def test_explicit_redesign_has_scoped_cas_reason_and_new_design_evidence(tmp_path):
    class Redesign(Session):
        def next_action(self, context, allowed, **kwargs):
            if (context['stage'] == Stage.DEBUG.value and context['task']['iterations']['design'] == 1
                    and not any(r['name'] == 'request_redesign' for r in context['evidence'])):
                return 'request_redesign', dict(reason='Replace unstable algorithm after numeric evidence')
            return super().next_action(context, allowed, **kwargs)
    runner, _, database = make(tmp_path, Redesign())
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        requests = [e for e in runner.events if e['type'] == 'project-tool' and e['name'] == 'request_redesign']
        assert len(requests) == 1
        request = requests[0]
        proof = runner.store.read_json(request['reference'])
        assert proof['stage'] == Stage.DEBUG.value
        assert proof['contract_ref'] == runner.contract_ref
        step = next(r for r in result['history'] if r.get('evidence', {}).get('operation_id') == proof['operation_id'])
        assert step['redesign'] is True
        assert request['reference'] in step['evidence']['artifact_refs']
        assert runner.workflow.state['design'] == 2
        from rsi.runtime import WorkflowError
        with pytest.raises(WorkflowError, match='redesign requires'):
            runner.tool('request_redesign', Stage.ACCEPTANCE, {'reason': 'skip stage'}, 'unknown')
    finally:
        database.close()


@pytest.mark.parametrize('requested_path', ['guidance.md', '../guidance.md'])
def test_readonly_path_error_is_durable_feedback_and_model_corrects(tmp_path, requested_path):
    class CorrectPath(Session):
        def next_action(self, context, allowed, **kwargs):
            if context['stage'] == Stage.TASK_CONSTRUCTION.value:
                rows = context['evidence']
                if not any(r['name'] == 'query_kdb' for r in rows):
                    return 'query_kdb', dict(query='stable sum')
                reads = [r['result'] for r in rows if r['name'] == 'read_project_file']
                if not reads:
                    return 'read_project_file', dict(path=requested_path)
                assert reads[0]['status'] == 'error'
                assert reads[0]['evidence']['path'] == requested_path
                assert 'content' not in reads[0]['evidence']
                if len(reads) == 1:
                    assert context['source']['path'] == 'solution.py'
                    return 'read_project_file', dict(path='helper.py')
                assert reads[1]['status'] == 'ok'
                assert reads[1]['evidence']['content'] == 'value = 1\n'
                # Select solution again before numeric debug needs to edit it.
                if len(reads) == 2:
                    return 'read_project_file', dict(path='solution.py')
            return super().next_action(context, allowed, **kwargs)
    runner, _, database = make(tmp_path, CorrectPath())
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        errors = [e for e in runner.events if e['type'] == 'project-tool' and e['status'] == 'error']
        assert len(errors) == 1
        payload = runner.store.read_json(errors[0]['reference'])
        assert payload['error_type'] in ('ValueError', 'FileNotFoundError')
        assert payload['editable_files'] == ['solution.py', 'helper.py']
        assert result['pending'] is None
    finally:
        database.close()


def test_readonly_recovery_does_not_swallow_other_errors(tmp_path, monkeypatch):
    runner, _, database = make(tmp_path)
    try:
        def denied(_):
            raise PermissionError('unreadable protected input')
        monkeypatch.setattr(runner.backend, 'read', denied)
        with pytest.raises(PermissionError):
            runner.tool('read_project_file', Stage.PREFLIGHT, dict(path='solution.py'), 'unused')
        assert runner.events == []
    finally:
        database.close()


def test_evidence_inventory_is_compact_and_full_records_remain_available(tmp_path):
    import json
    runner, _, database = make(tmp_path)
    try:
        runner.actor.step(runner.context())
        runner.actor.step(runner.context())
        operation = runner.workflow.begin()
        callback = runner.service.query_backend
        def with_large_manifest(payload):
            response = callback(payload)  # Real configured KnowledgeDB query.
            response['provenance']['test_large_manifest'] = 'x' * 10000
            return response
        runner.service.query_backend = with_large_manifest
        retrieval = runner.tool('query_kdb', Stage.DESIGN, dict(query='stable sum'), operation)
        summary = retrieval['summary']
        full = runner.store.read_json(retrieval['reference'])
        assert retrieval['evidence'] == full
        assert summary['query_ref'] == retrieval['reference']
        assert summary['operation_id'] == operation and summary['stage'] == Stage.DESIGN.value
        assert summary['host_perform_ready'] is False
        assert summary['mapped_skill_reads_required'] == list(runner.bundle.skills_for(Stage.DESIGN))
        for group in ('results', 'negative_records'):
            count = len(full['payload']['response'][group])
            assert summary['counts'][group] == count
            assert summary['decision_indexes'][group] == list(range(count))
        assert len(json.dumps(summary)) < 2048
        assert len(full['payload']['response']['provenance']['test_large_manifest']) == 10000
        retro = runner.tool('retrospect', Stage.DESIGN, dict(supplement='Consider actual history'), operation)
        payload = runner.store.read_json(retro['reference'])['payload']
        assert retro['summary']['retrospective_ref'] == retro['reference']
        assert retro['summary']['history_count'] == len(payload['history']) == 2
        assert retro['summary']['failure_indexes'] == payload['failures']
        assert retro['summary']['redirection_indexes'] == payload['redirections']
        assert retro['evidence']['payload'] == payload
    finally:
        runner.actor.close()
        database.close()
