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


def make(tmp_path, session=None, *, config_overrides=None, budgets=None, **overrides):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'solution.py').write_text('def solve(xs):\n    total = 0.0\n    for x in xs: total += x\n    return total\n')
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
    runner, _, database = make(tmp_path, config_overrides=dict(
        performance_seconds=1e-9,
        commands={
            'compile': [sys.executable, '-m', 'py_compile', 'solution.py'],
            'test': [sys.executable, '{verifier}'],
            'validation': [sys.executable, '{verifier}'],
            'timing': [sys.executable, 'solution.py']}))
    try:
        result = runner.run()
        assert result['status'] == 'rejected', result
        assert result['correctness'] is True
        assert result['performance'] == 'shortfall'
        (runner.backend.project / 'solution.py').write_text('invalid source!')
        reopened = runner.run()
        assert reopened['status'] == 'suspended'
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
