"""Production composition with scripted decisions and real CPU/KDB services."""
from rsi.event_journal import read_event_refs
from core.artifacts import canonical_bytes

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


def make(tmp_path, session=None, *, config_overrides=None, budgets=None, source_text=None,
         extra_files=None, **overrides):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text(source_text or 'def solve(xs):\n    total = 0.0\n    for x in xs: total += x\n    return total\n')
    (source / 'helper.py').write_text('value = 1\n')
    for name, content in (extra_files or {}).items():
        (source / name).write_bytes(content)
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


def test_noop_with_redundant_edit_receives_feedback_before_host_dispatch(tmp_path):
    class RedundantEdit(Session):
        attempted = False
        def next_action(self, context, allowed, **kwargs):
            name, args = super().next_action(context, allowed, **kwargs)
            if name == 'perform' and context['stage'] == 'optimisation' and not self.attempted:
                self.attempted = True
                source = context['source']
                args = dict(args, edit=dict(path=source['path'], sha256=source['sha256'],
                                            source=source['text']))
            return name, args
    runner, _, database = make(tmp_path, RedundantEdit())
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        invalid = [e for e in runner.events if e['type']=='actor-invalid-action']
        assert len(invalid) == 1 and invalid[0]['accepted'] is False
        operation = invalid[0]['operation_id']
        assert len([e for e in runner.events if e['type']=='host-dispatch'
                    and e['operation_id']==operation]) == 1
        assert not any(e['type']=='source-edit' and e['operation_id']==operation for e in runner.events)
        assert runner.workflow.accounting.seconds('thinking') > 0
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
        proof = runner.terminal_failure_proof()
        assert proof['actor_seconds_total'] == 0
        assert proof['pending'] is None
    finally:
        database.close()


@pytest.mark.parametrize('error_type,classification', [
    ('PiProviderError', 'provider_failure'),
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
        refs = read_event_refs(runner.root)
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


@pytest.mark.parametrize(('requested_path', 'error_type'), [
    ('guidance.md', 'FileNotFoundError'), ('../guidance.md', 'ValueError'),
    ('.', 'IsADirectoryError'), ('solution.py/child', 'NotADirectoryError'),
    ('artifact.bin', 'UnicodeDecodeError')])
def test_readonly_path_error_is_durable_feedback_and_model_corrects(tmp_path, requested_path, error_type):
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
                assert reads[0]['evidence']['error_type'] == error_type
                assert reads[0]['evidence']['selected_file'] == 'solution.py'
                assert reads[0]['evidence']['editable_files'] == ['solution.py', 'helper.py']
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
    runner, _, database = make(tmp_path, CorrectPath(), extra_files={'artifact.bin': b'\xff\xfe\x00'})
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        errors = [e for e in runner.events if e['type'] == 'project-tool' and e['status'] == 'error']
        assert len(errors) == 1
        payload = runner.store.read_json(errors[0]['reference'])
        assert payload['error_type'] == error_type
        assert payload['editable_files'] == ['solution.py', 'helper.py']
        assert result['pending'] is None
    finally:
        database.close()


@pytest.mark.parametrize('failure', [PermissionError('unreadable protected input'), OSError(5, 'backend I/O failed')])
def test_readonly_recovery_does_not_swallow_other_errors(tmp_path, monkeypatch, failure):
    runner, _, database = make(tmp_path)
    try:
        def denied(_):
            raise failure
        monkeypatch.setattr(runner.backend, 'read', denied)
        with pytest.raises(type(failure)) as caught:
            runner.tool('read_project_file', Stage.PREFLIGHT, dict(path='solution.py'), 'unused')
        assert caught.value is failure
        assert runner.selected_file == 'solution.py'
        assert runner.events == []
    finally:
        database.close()


@pytest.mark.parametrize(('path', 'error_type'), [('.', 'IsADirectoryError'),
    ('helper.py/child', 'NotADirectoryError'), ('artifact.bin', 'UnicodeDecodeError')])
def test_invalid_read_preserves_nondefault_selection_and_next_valid_read(tmp_path, path, error_type):
    runner, _, database = make(tmp_path, extra_files={'artifact.bin': b'\xff\x00'})
    try:
        runner.tool('read_project_file', Stage.DESIGN, dict(path='helper.py'), 'read-fixture')
        failed = runner.tool('read_project_file', Stage.DESIGN, dict(path=path), 'read-fixture')
        assert failed['status'] == 'error' and failed['evidence']['error_type'] == error_type
        assert failed['evidence']['selected_file'] == runner.selected_file == 'helper.py'
        event = runner.events[-1]
        assert event['status'] == 'error' and event['reference'] == failed['reference']
        assert runner.store.read_json(event['reference']) == failed['evidence']
        valid = runner.tool('read_project_file', Stage.DESIGN, dict(path='solution.py'), 'read-fixture')
        assert valid['status'] == 'ok'
        assert valid['evidence']['content'] == (runner.backend.project / 'solution.py').read_text()
        assert runner.selected_file == 'solution.py'
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


def test_repeated_malformed_actions_recover_until_original_actor_budget(tmp_path):
    from coding.pi.session import ActionError
    clock = Clock()
    class Malformed(Session):
        def next_action(self, *args, **kwargs):
            self.calls += 1
            clock.value += 2
            raise ActionError('select_strategy missing required query_ref')
    session = Malformed()
    runner, injections, database = make(tmp_path, session, clock=clock,
                                       budgets=BudgetLimits(actor_seconds=5))
    try:
        result = runner.run()
        assert result['status'] == 'budget_exhausted', result
        assert runner.workflow.accounting.seconds('thinking') == 6
        assert session.calls == 3 and result['pending']
        assert runner.workflow.outcome is None
        proof = runner.terminal_failure_proof()
        assert proof['classification'] == 'budget_exhausted' and proof['actor_seconds_total'] == 6
        assert proof['pending'] == result['pending']
        feedback = [e for e in runner.events if e['type'] == 'actor-invalid-action']
        assert len(feedback) == 2 and all(e['accepted'] is False for e in feedback)
        assert not any(e['type'] == 'host-dispatch' for e in runner.events)
        reopened = ProjectRunner(runner.root, **injections).run()
        assert reopened['pending'] == result['pending'] and session.calls == 3
    finally:
        database.close()


def test_malformed_model_arguments_and_missing_read_are_corrected_in_same_operation(tmp_path):
    from coding.pi.session import ActionError
    class CorrectArguments(Session):
        rejected = False
        observed_feedback = False
        def next_action(self, context, allowed, **kwargs):
            if context['stage'] == Stage.TASK_CONSTRUCTION.value:
                if not self.rejected:
                    self.rejected = True
                    raise ActionError('read_project_file requires path')
                if not self.observed_feedback:
                    assert context['observation']['event'] == 'invalid-action'
                    assert context['observation']['accepted'] is False
                    self.observed_feedback = True
                    return 'read_project_file', dict(path='guidance.md')
                reads = [r for r in context['evidence'] if r['name'] == 'read_project_file']
                if len(reads) == 1:
                    assert reads[0]['result']['status'] == 'error'
                    return 'read_project_file', dict(path='solution.py')
            return super().next_action(context, allowed, **kwargs)
    session = CorrectArguments()
    runner, _, database = make(tmp_path, session)
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        assert session.observed_feedback
        failures = [e for e in runner.events if e['type'] == 'actor-invalid-action']
        assert len(failures) == 1
        completed = [r for r in result['history'] if r.get('evidence', {}).get('operation_id')
                     == failures[0]['operation_id']]
        assert len(completed) == 1 and completed[0]['passed']
    finally:
        database.close()


@pytest.mark.parametrize('declared', [True, False])
def test_generated_output_configuration_controls_source_identity(tmp_path, declared):
    compile_script = (
        "import py_compile; from pathlib import Path; "
        "py_compile.compile('solution.py', doraise=True); "
        "Path('results.json').write_text('{}'); "
        "Path('build').mkdir(exist_ok=True); Path('build/coverage.json').write_text('{}')")
    runner, injections, database = make(tmp_path,
        source_text='from math import fsum\ndef solve(xs): return fsum(xs)\n',
        config_overrides=dict(generated_paths=['results.json', 'build'] if declared else [],
            commands={'compile': [sys.executable, '-c', compile_script],
                      'test': [sys.executable, '{verifier}'],
                      'validation': [sys.executable, '{verifier}']}))
    try:
        initial = runner.backend.project_hash()
        result = runner.run()
        assert (runner.backend.project / 'results.json').read_text() == '{}'
        assert (runner.backend.project / 'build/coverage.json').read_text() == '{}'
        assert runner.config.generated_paths == (('results.json', 'build') if declared else ())
        if declared:
            assert result['status'] == 'accepted', result
            assert result['source_hash'] == initial
            assert runner.backend.contract['generated_paths'] == ['build', 'results.json']
            reopened = ProjectRunner(runner.root, **injections)
            assert reopened.config.generated_paths == runner.config.generated_paths
            assert reopened.backend.project_hash() == initial
            (reopened.backend.project / 'unlisted.json').write_text('{}')
            assert reopened.backend.project_hash() != initial
        else:
            assert result['status'] == 'suspended', result
            assert result['source_hash'] != initial
            assert runner.workflow.outcome is None
            rows = runner.latest('command-results')['receipts']
            assert rows[0]['classification'] == 'source_changed'
            proof = runner.terminal_failure_proof()
            assert proof['source_hash'] == result['source_hash']
            assert proof['stage_initial_source'] == initial != proof['source_hash']
    finally:
        database.close()


def failed_after_query(tmp_path, *, after_commands=False):
    from coding.pi.rpc import PiDeadlineError
    class Terminal(Session):
        def next_action(self, context, allowed, **kwargs):
            if (context['stage'] == (Stage.DEBUG.value if after_commands else Stage.DESIGN.value)
                    and (after_commands or any(r['name'] == 'query_kdb' for r in context['evidence']))):
                raise PiDeadlineError('observed terminal provider deadline')
            return super().next_action(context, allowed, **kwargs)
    runner, injections, database = make(tmp_path, Terminal())
    result = runner.run()
    assert result['status'] == 'suspended' and result['actor_closure_ref']
    return runner, injections, database, result


@pytest.mark.parametrize('after_commands', [False, True])
def test_terminal_failure_proof_preserves_pending_and_completed_effects(tmp_path, after_commands):
    import json
    runner, injections, database, result = failed_after_query(tmp_path, after_commands=after_commands)
    try:
        before_state = (runner.root / 'workflow.json').read_bytes()
        before_index = canonical_bytes(read_event_refs(runner.root))
        before_objects = set(runner.store.objects.rglob('*'))
        proof = runner.terminal_failure_proof(closure_ref=result['actor_closure_ref'])
        assert proof['outcome'] == 'failed' and proof['classification'] == 'provider_timeout'
        assert proof['pending'] == result['pending']
        assert proof['stage_completion_claim'] is False and proof['replay_allowed'] is False
        assert proof['retrieval_receipts']
        assert len(proof['command_receipts']) == (2 if after_commands else 0)
        assert proof['actor_seconds_total'] == runner.workflow.accounting.seconds()
        assert before_state == (runner.root / 'workflow.json').read_bytes()
        assert before_index == canonical_bytes(read_event_refs(runner.root))
        assert before_objects == set(runner.store.objects.rglob('*'))
        reopened = ProjectRunner(runner.root, **injections)
        assert reopened.terminal_failure_proof() == proof
        reopened.run()  # Observing the retained pending work never replays it.
        assert reopened.terminal_failure_proof() == proof
        assert reopened.workflow.outcome is None and reopened.workflow.pending == proof['pending']
    finally:
        database.close()


def test_unobserved_restart_requires_explicit_terminal_observation(tmp_path):
    from rsi.runtime import WorkflowError
    runner, injections, database = make(tmp_path)
    try:
        runner.actor.step(runner.context())
        operation = runner.workflow.begin()
        runner.workflow.dispatched(operation, 'pi:actually-closed-test-session')
        # The host observed this test session close; reopening alone has no receipt.
        session = Session()
        session.close()
        reopened = ProjectRunner(runner.root, **injections)
        result = reopened.run()
        with pytest.raises(WorkflowError, match='closure receipt'):
            reopened.terminal_failure_proof()
        pending = dict(reopened.workflow.pending)
        controller_proof = dict(handle=pending['handle'], actor_closed=True, provider_settled=True,
            observation='Host observed the owned test session close without running commands/queries',
            completed_query_refs=[])
        reopened.record_terminal_observation(result, controller_proof=controller_proof,
                                              observed_source_hash=reopened.backend.project_hash())
        proof = reopened.terminal_failure_proof()
        assert proof['pending'] == pending and reopened.workflow.pending == pending
        assert proof['command_receipts'] == [] and proof['retrieval_receipts'] == []
    finally:
        database.close()


def test_failure_proof_refuses_pending_execution_reservation(tmp_path, monkeypatch):
    from rsi.runtime import WorkflowError
    runner, _, database = make(tmp_path)
    def disconnected(operation, *args, **kwargs):
        runner.workflow.reserve_execution(operation)
        raise WorkflowError('execution disconnected before receipt')
    monkeypatch.setattr(runner.backend, 'run', disconnected)
    try:
        result = runner.run()
        assert result['status'] == 'suspended' and result['actor_closure_ref']
        with pytest.raises(WorkflowError, match='complete receipt'):
            runner.terminal_failure_proof()
        assert runner.workflow.pending == result['pending']
    finally:
        database.close()


def test_unknown_query_cannot_admit_failure(tmp_path):
    from rsi.runtime import WorkflowError
    def unavailable(_):
        raise OSError('query lifecycle unknown')
    runner, _, database = make(tmp_path, query=unavailable)
    try:
        result = runner.run()
        assert result['status'] == 'suspended'
        with pytest.raises(WorkflowError, match='query'):
            runner.terminal_failure_proof()
    finally:
        database.close()


def test_failure_proof_refuses_changed_source_manifest_and_clock(tmp_path):
    from rsi.runtime import WorkflowError
    import json
    runner, _, database, _ = failed_after_query(tmp_path)
    try:
        path = runner.backend.project / 'solution.py'
        original = path.read_text()
        path.write_text(original + '\n# unexplained change\n')
        with pytest.raises(WorkflowError):
            runner.terminal_failure_proof()
        path.write_text(original)
        manifest = runner.root / 'manifest.json'
        original_manifest = manifest.read_bytes()
        manifest.write_text(json.dumps({'contract': 'changed'}))
        with pytest.raises(WorkflowError, match='manifest'):
            runner.terminal_failure_proof()
        manifest.write_bytes(original_manifest)
        token = runner.workflow.accounting.start('execution')
        with pytest.raises(WorkflowError):
            runner.terminal_failure_proof()
        runner.workflow.accounting.end(token)
        assert runner.workflow.pending is not None
    finally:
        database.close()


def test_unknown_session_close_does_not_certify_production_provider_settled(tmp_path):
    from coding.pi.rpc import PiDeadlineError
    from rsi.runtime import WorkflowError
    class Unobserved(Session):
        def next_action(self, *args, **kwargs):
            raise PiDeadlineError('provider lifecycle unobserved')
    runner, _, database = make(tmp_path, Unobserved(), test_backend=False)
    try:
        result = runner.run()
        assert result['status'] == 'suspended'
        with pytest.raises(WorkflowError, match='closure was not observed'):
            runner.terminal_failure_proof()
    finally:
        database.close()


@pytest.mark.parametrize('error', [OSError('dependency readiness failed'), TimeoutError('readiness timed out')])
def test_prepare_failure_retains_closed_session_proof_before_dispatch(tmp_path, error):
    class Startup(Session):
        closed = 0
        def prepare(self, timeout, *, schemas):
            raise error
        def close(self):
            self.closed += 1
    session = Startup()
    runner, injections, database = make(tmp_path, session)
    try:
        result = runner.run()
        assert result['status'] == 'suspended' and result['actor_closure_ref']
        assert result['pending']['handle'] is None and session.closed == 1 and session.calls == 0
        proof = runner.terminal_failure_proof()
        assert proof['outcome'] == 'failed' and proof['pending'] == result['pending']
        assert proof['actor_seconds_total'] == 0
        reopened = ProjectRunner(runner.root, **injections)
        assert reopened.terminal_failure_proof() == proof
        assert runner.workflow.pending == result['pending'] and runner.workflow.outcome is None
    finally:
        database.close()


def test_real_rpc_setup_close_and_provider_settlement_are_observed(tmp_path):
    from pathlib import Path
    from threading import Event
    from types import SimpleNamespace
    from coding.pi.rpc import PiRPC
    from rsi.actor import PiSessionAdapter
    class StartingPi(PiSessionAdapter):
        def __init__(self):
            rpc = PiRPC([sys.executable, '-c', 'import sys; sys.stdin.readline()'],
                        cwd=Path(tmp_path), env=None)
            settled = Event()
            settled.set()  # No provider request was dispatched during failed setup.
            self.session = SimpleNamespace(rpc=rpc, gateway=SimpleNamespace(finished=settled), close=rpc.close)
        def prepare(self, timeout, *, schemas):
            raise TimeoutError('handshake readiness timeout')
    sessions = []
    def factory():
        session = StartingPi()
        sessions.append(session)
        return session
    runner, _, database = make(tmp_path, session_factory=factory, test_backend=False)
    try:
        result = runner.run()
        assert sessions[0].session.rpc.process.poll() is not None
        assert result['pending']['handle'] is None
        closure = runner.store.read_json(result['actor_closure_ref'])['observation']
        assert closure['actor_closed'] is True and closure['provider_settled'] is True
        assert runner.terminal_failure_proof()['outcome'] == 'failed'
    finally:
        for session in sessions:
            session.session.rpc.close()
        database.close()


def test_raising_factory_partial_start_needs_scoped_explicit_observation(tmp_path):
    import subprocess
    from rsi.runtime import WorkflowError
    processes = []
    def factory():
        processes.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        raise OSError('factory raised after partially starting a child')
    runner, injections, database = make(tmp_path, session_factory=factory)
    try:
        result = runner.run()
        assert result['actor_closure_ref'] is None and result['pending']['handle'] is None
        assert processes[0].poll() is None
        with pytest.raises(WorkflowError, match='closure receipt'):
            runner.terminal_failure_proof()
        reopened = ProjectRunner(runner.root, **injections)
        observed = reopened.run()
        assert len(processes) == 1 and observed['actor_closure_ref'] is None
        # The host now actually terminates/reaps the child it owns. No provider ran.
        processes[0].terminate()
        processes[0].wait(timeout=3)
        pending = reopened.workflow.pending
        proof = dict(operation_id=pending['id'], handle=None, no_actor_dispatched=True,
            actor_closed=True, provider_settled=True,
            observation='Host terminated/reaped its partially started child; no Pi/provider dispatch occurred')
        with pytest.raises(WorkflowError):
            reopened.record_terminal_observation(observed, controller_proof=dict(proof, operation_id='other'),
                                                  observed_source_hash=reopened.backend.project_hash())
        reopened.record_terminal_observation(observed, controller_proof=proof,
                                              observed_source_hash=reopened.backend.project_hash())
        admitted = reopened.terminal_failure_proof()
        assert admitted['pending'] == pending and reopened.workflow.pending == pending
        # The narrow no-effect API also permits explicitly observed None handles,
        # but releases only after reject_operation and keeps consumed allocations.
        failure = reopened.record_admission_failure(pending['id'],
            controller_proof=dict(proof, no_host_effects=True), observed_source_hash=reopened.backend.project_hash())
        charged, limits = reopened.workflow.accounting.seconds(), dict(reopened.workflow.state['limits'])
        reopened.workflow.reject_operation(pending['id'], failure, reopened.backend.project_hash())
        assert reopened.workflow.pending is None
        assert reopened.workflow.accounting.seconds() == charged and reopened.workflow.state['limits'] == limits
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        database.close()


def test_unfinished_query_dispatch_vetoes_admission_record_and_old_rejection(tmp_path):
    from rsi.runtime import WorkflowError
    runner, injections, database = make(tmp_path)
    try:
        runner.actor.step(runner.context())
        runner.actor.step(runner.context())
        operation = runner.workflow.begin()
        runner.workflow.dispatched(operation, 'pi:closed-before-observation')
        runner.actor.close()
        source = runner.backend.project_hash()
        proof = dict(handle=runner.workflow.pending['handle'], actor_closed=True, no_host_effects=True,
                     observation='Host observed owned Pi closed; callback status must be checked separately')
        old_reference = runner.record_admission_failure(operation, controller_proof=proof, observed_source_hash=source)
        token = runner.workflow.accounting.start('retrieval')
        runner.append(dict(type='query-dispatch', operation_id=operation, call_id='unknown-callback',
                           request_ref=runner.store.put_json(dict(query='stable sum'))))
        observed_end = runner.workflow.clock()
        # Crash/reopen: reconcile only observed elapsed time, not callback completion.
        reopened = ProjectRunner(runner.root, **injections)
        reopened.workflow.accounting.reconcile(token, observed_end)
        assert reopened.no_host_effects(operation) is False
        with pytest.raises(WorkflowError, match='no host effects'):
            reopened.record_admission_failure(operation, controller_proof=proof, observed_source_hash=source)
        with pytest.raises(WorkflowError, match='untrusted'):
            reopened.workflow.reject_operation(operation, old_reference, source)
        assert reopened.workflow.pending['id'] == operation
    finally:
        database.close()


@pytest.mark.parametrize('correction', ['redirect', 'reject'])
def test_strategy_rejection_reuses_real_receipts_and_original_budget(tmp_path, correction):
    from copy import deepcopy
    import time

    class ModelClock:
        value = 0.

        def __call__(self):
            return time.time() + self.value

    clock = ModelClock()

    class CorrectStrategy(Session):
        invalid_args = None
        corrected_args = None
        feedback = None

        def next_action(self, context, allowed, **kwargs):
            clock.value += 2
            if self.invalid_args is not None and self.corrected_args is None:
                self.feedback = deepcopy(context)
                assert not any(e['type'] == 'strategy-evidence' and e['kind'] == 'selection'
                               for e in runner.events)
                assert not any(row['name'] == 'select_strategy' for row in context['evidence'])
                assert context['observation']['event'] == 'invalid-action'
                assert context['observation']['action'] == 'select_strategy'
                assert context['observation']['accepted'] is False
                args = deepcopy(self.invalid_args)
                for decision in args['decisions']:
                    if decision['group'] == 'negative_records':
                        if correction == 'redirect':
                            decision['redirection'] = 'Replace naive accumulation with fsum'
                        else:
                            decision['action'] = 'rejected'
                self.corrected_args = args
                return 'select_strategy', args
            name, args = super().next_action(context, allowed, **kwargs)
            if name == 'select_strategy' and self.invalid_args is None:
                for decision in args['decisions']:
                    if decision['group'] == 'negative_records':
                        decision['action'] = 'used'
                        decision.pop('redirection', None)
                self.invalid_args = deepcopy(args)
            return name, args

    session = CorrectStrategy()
    runner, _, database = make(tmp_path, session, clock=clock,
        source_text='from math import fsum\ndef solve(xs): return fsum(xs)\n')
    try:
        result = runner.run()
        assert result['status'] == 'accepted', result
        feedback = [e for e in runner.events if e['type'] == 'actor-invalid-action']
        assert len(feedback) == 1
        failure = feedback[0]
        operation = failure['operation_id']
        tools = [e for e in runner.events if e['type'] == 'project-tool'
                 and e['operation_id'] == operation]
        assert [e['name'] for e in tools] == ['query_kdb', 'retrospect', 'select_strategy']
        payload = runner.store.read_json(tools[-1]['reference'])['payload']
        assert payload == session.corrected_args
        for field in ('query_ref', 'retrospective_ref', 'skill_refs'):
            assert payload[field] == session.invalid_args[field]
        inventory = session.feedback['evidence_receipts']
        assert {row['reference'] for row in inventory} == {e['reference'] for e in tools[:2]}
        assert {row['receipt_id'] for row in session.feedback['skill_receipts']} == set(payload['skill_refs'])
        completed = [row for row in result['history'] if row['evidence']['operation_id'] == operation]
        assert len(completed) == 1 and completed[0]['passed']
        stages = [row['evidence']['stage'] for row in result['history']]
        assert sorted(stages) == sorted(stage.value for stage in Stage)
        assert runner.workflow.accounting.seconds('thinking') >= session.calls * 2 + 2
    finally:
        database.close()


def test_trusted_executor_binding_is_frozen_across_create_and_reopen(tmp_path):
    from rsi.python_execution import CommandResult

    identities, commands = [], []

    class Executor:
        def execute(self, request, *, observe, remaining):
            commands.append(request)
            observe('command', handle='remote:test:job:retained')
            assert remaining() > 0
            return CommandResult(0)

    def factory(identity):
        identities.append(identity)
        return Executor()

    runner, injections, database = make(tmp_path,
        config_overrides={'executor_identity': 'test-runtime@revision'},
        executor_factory=factory, budgets=BudgetLimits(actor_seconds=10))
    try:
        receipt = runner.backend.run('injected', 'compile')
        reopened = ProjectRunner(runner.root, **injections)
        assert identities == ['test-runtime@revision', 'test-runtime@revision']
        assert reopened.config.executor_identity == 'test-runtime@revision'
        assert reopened.backend.run('injected', 'compile') == receipt
        assert len(commands) == 1
        missing_factory = {key: value for key, value in injections.items() if key != 'executor_factory'}
        with pytest.raises(ValueError, match='trusted executor factory'):
            ProjectRunner(runner.root, **missing_factory)
    finally:
        database.close()


def test_local_project_does_not_construct_optional_executor(tmp_path):
    def factory(identity):
        pytest.fail('local project must not construct an executor')

    runner, _, database = make(tmp_path, executor_factory=factory)
    try:
        receipt = runner.backend.run('local-compile', 'compile')
        assert receipt['classification'] == 'passed'
        assert receipt['pid'] > 0
        assert 'executor_identity' not in receipt
    finally:
        database.close()
