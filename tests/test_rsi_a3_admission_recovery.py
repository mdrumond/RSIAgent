"""Official A3 factory, observed CPU failures and elapsed-preserving retry admission."""
from dataclasses import asdict
import json

import pytest

from rsi.domain_bindings import a3_adoption
from rsi.a3_recovery import observed_prelaunch_root


def official(owner, session):
    from benchmarks.a3kernels.knowledge_agent import KnowledgeQuery
    baseline = owner.baseline_knowledge
    def query(payload):
        return dict(results=[asdict(row) for row in baseline.query(KnowledgeQuery(payload['query']))],
                    negative_records=[], provenance=dict(manifest=json.loads(baseline.manifest.to_json())))
    query.canonical_identity = lambda: dict(manifest=json.loads(baseline.manifest.to_json()))
    return a3_adoption(baseline_query=query, session_factory=lambda: session,
                       test_backend=True, clock=owner.clock)


@pytest.mark.parametrize('stage,error_type,safe', [
    ('prepare','RuntimeError',True), ('compile','AmbiguousRemoteOutcome',False),
])
def test_official_a3_carrier_retains_failure_and_only_observed_prelaunch_can_retry(tmp_path,stage,error_type,safe):
    from canonical_a3_fixtures import make_trial
    from benchmarks.a3kernels.phase1_protocol import FailedEvidence
    from benchmarks.a3kernels.project_execution import RecoveryStarterEvidenceUnavailable
    from test_a3_trial_loop import FakeCandidate, Resolver
    now=[1000.]
    class Candidate(FakeCandidate):
        calls=0
        def compile(self, source, workdir, **options):
            self.calls+=1;now[0]+=2
            from benchmarks.a3kernels.candidate import A3CandidateBackend
            plan=A3CandidateBackend().plan(source, **options)
            raise RecoveryStarterEvidenceUnavailable(FailedEvidence.create(
                plan,stage=stage,error_type=error_type,detail='Actual CPU fixture observed failure'))
    candidate=Candidate(Resolver())
    owner,session=make_trial(tmp_path/'first',candidate=candidate,clock=lambda:now[0])
    owner.canonical=official(owner,session)
    result=owner.run()
    root=owner.workdir/'canonical'
    assert result.status=='infrastructure-unverified' and candidate.calls==1 and result.failures
    assert observed_prelaunch_root(root) is safe
    if safe:
        original=json.loads((root/'workflow.json').read_text())
        fresh,next_session=make_trial(tmp_path/'next',clock=lambda:now[0])
        fresh.canonical_artifact_root=owner.workdir/'canonical-artifacts'
        fresh.canonical_prior_root=root
        fresh.canonical=official(fresh,next_session)
        runner=fresh.canonical.build(fresh)
        imported=runner.workflow.state['intervals']
        assert imported and all(row[2] is not None for row in imported.values())
        assert runner.workflow.accounting.seconds()==2
        assert runner.workflow.limits.actor_seconds==original['limits']['actor_seconds']
        assert json.loads((root/'workflow.json').read_text())==original
        assert runner.run()['status']=='accepted'
        assert runner.workflow.accounting.seconds()==2
    else:
        frozen=(root/'workflow.json').read_bytes()
        assert owner.run().status=='suspended'
        assert candidate.calls==1 and (root/'workflow.json').read_bytes()==frozen


@pytest.mark.parametrize('status',['budget-exhausted','budget_exhausted','suspended'])
def test_actual_trial_status_roundtrip_preserves_wire_and_telemetry(tmp_path,status):
    from benchmarks.a3kernels.live_composition import LiveComposition
    from benchmarks.a3kernels.phase1_evidence import EvidenceLedger
    from benchmarks.a3kernels.phase1_wave import foundation_cells
    from benchmarks.a3kernels.phase1_registry import DEFAULT_PROPOSALS
    from benchmarks.a3kernels.trial import TrialResult
    evidence=EvidenceLedger(tmp_path/'evidence.jsonl')
    trial=TrialResult(status,3,400000,None,None,2,('retained-failure',),None)
    LiveComposition._append_attempt_outcome(evidence,cell=foundation_cells()[0],
        proposal=DEFAULT_PROPOSALS[0],execution_profile='gz-a3',ordinal=0,trial=trial)
    restored=LiveComposition._trial_from_outcome(evidence.entries[-1].payload['infrastructure_retry'])
    assert restored==trial


@pytest.mark.parametrize('operation',['compile','run'])
@pytest.mark.parametrize('typed',[True,False],ids=['actual-source-validator','arbitrary-value-error'])
def test_source_admission_repairs_only_authentic_typed_validation(tmp_path,operation,typed):
    from canonical_a3_fixtures import make_trial, A3Session
    from test_a3_trial_loop import FakeCandidate, Resolver, SOURCE
    from benchmarks.a3kernels.candidate import validate_candidate_source
    malformed=SOURCE.replace('vector_add','wrong_signature')
    calls=[]
    class Candidate(FakeCandidate):
        def invoke(self, name, source, workdir, **options):
            calls.append((name,source))
            if name==operation and source==malformed:
                if typed:
                    validate_candidate_source(source)  # Existing validator, not a fabricated typed exception.
                raise ValueError('Callback failed without a no-effects admission contract')
            return getattr(super(),name)(source,workdir,**options)
        def compile(self,source,workdir,**options):
            return self.invoke('compile',source,workdir,**options)
        def run(self,source,workdir,**options):
            return self.invoke('run',source,workdir,**options)
    owner,session=make_trial(tmp_path, candidate=Candidate(Resolver()),
        session=A3Session(initial=malformed,repaired=SOURCE))
    owner.canonical=official(owner,session)
    result=owner.run()
    root=owner.workdir/'canonical'
    state=json.loads((root/'workflow.json').read_text())
    if typed:
        assert result.status=='passed' and result.verified is not None and result.verified.passed
        assert any(name==operation and source==malformed for name,source in calls)
        assert any(name=='run' and source!=malformed for name,source in calls)
        assert len(owner.memory.read())==1 and result.failures
        runner=owner.canonical.build(owner)
        failures=[runner.store.read_json(event['reference']) for event in runner.events
            if event['type']=='domain-result' and runner.store.read_json(event['reference'])['status']=='failed']
        assert len(failures)==1 and failures[0]['observed_terminal'] is True
        from rsi.domain_checkpoint import decode
        assert decode(failures[0]['result'])['error_type']=='CandidateSourceAdmissionError'
        assert any(row['passed'] is False for row in state['history'])
    else:
        assert result.status=='suspended' and state['pending'] is not None and not owner.memory.read()
        before=list(calls);raw=(root/'workflow.json').read_bytes()
        assert owner.run().status=='suspended'
        assert calls==before and (root/'workflow.json').read_bytes()==raw


def test_observed_source_admission_failure_cannot_satisfy_passing_test(tmp_path,monkeypatch):
    from dataclasses import replace
    from canonical_a3_fixtures import make_trial, A3Session
    from test_a3_trial_loop import FakeCandidate, Resolver, SOURCE
    from benchmarks.a3kernels.candidate import validate_candidate_source
    from rsi.domain_bindings import A3Host
    from rsi.contracts import Stage
    calls=[]
    class Candidate(FakeCandidate):
        def compile(self,source,workdir,**options):
            calls.append(source)
            validate_candidate_source(source)
            return super().compile(source,workdir,**options)
    original=A3Host.execute
    def wrong_passing_claim(self,stage,*args):
        observation=original(self,stage,*args)
        if stage==Stage.TEST and observation.passed is False:
            return replace(observation,passed=True)
        return observation
    monkeypatch.setattr(A3Host,'execute',wrong_passing_claim)
    malformed=SOURCE.replace('vector_add','wrong_signature')
    owner,session=make_trial(tmp_path,candidate=Candidate(Resolver()),
        session=A3Session(initial=malformed,repaired=SOURCE))
    owner.canonical=official(owner,session)
    result=owner.run()
    state=json.loads((owner.workdir/'canonical/workflow.json').read_text())
    assert result.status=='suspended' and state['pending']['stage']=='test'
    assert calls==[malformed] and not owner.memory.read()
    assert not any(row['evidence']['stage']=='test' for row in state['history'])
