"""Native compaction fallback and actual terminal RPC outcomes; no paid provider."""
import json
import queue

import pytest

from coding.pi.rpc import PiCompactionError, PiRPC
from coding.pi.schema import STRING, obj
from rsi.actor import EvidenceTool, StageResult
from rsi.contracts import BudgetLimits, Stage
from rsi.skills import builtin_bundle
from test_rsi_pi_controller import native_actor, response


@pytest.mark.parametrize('ack_after_settlement', [False, True])
@pytest.mark.parametrize('recovered', [False, True])
def test_rpc_reports_unresolved_compaction_but_accepts_native_recovery(ack_after_settlement, recovered):
    rpc = object.__new__(PiRPC)
    rpc.timeout, rpc.events, rpc.records = 1, [], queue.Queue()

    def send(command):
        acknowledgement = dict(type='response', id=command['id'], success=True,
                               data={'disposition': 'started'})
        events = [dict(type='message_end', message=dict(role='assistant', stopReason='error',
                      errorMessage='413: context_length_exceeded')),
                  dict(type='compaction_start', reason='overflow'),
                  dict(type='compaction_end', reason='overflow', aborted=True, willRetry=False)]
        if recovered:
            events += [dict(type='compaction_end', reason='overflow', aborted=False),
                       dict(type='message_end', message=dict(role='assistant', stopReason='toolUse'))]
        events.append(dict(type='agent_settled', aborted=False))
        events.insert(len(events) if ack_after_settlement else 0, acknowledgement)
        for event in events:
            rpc.records.put(('stdout', json.dumps(event).encode()+b'\n'))

    rpc.send = send
    if recovered:
        rpc.prompt('continue the same operation')
        assert any(e['type']=='agent_settled' for e in rpc.events)
    else:
        with pytest.raises(PiCompactionError):
            rpc.prompt('continue the same operation')
        assert rpc.records.empty()


@pytest.mark.parametrize('fallback_succeeds', [False, True])
def test_actual_pi_native_fallback_preserves_history_effects_and_ledgers(monkeypatch, tmp_path, fallback_succeeds):
    events, requests, effects, summaries = [], [], [], []
    names = builtin_bundle().skills_for(Stage.TASK_CONSTRUCTION)
    partial = 'Incomplete checkpoint, deliberately stopped by the provider.'
    complete = 'Evidence receipts retained. Current task construction remains pending. Continue guarded tools.'

    def provider(*_, **kwargs):
        request = json.loads(kwargs['data'])
        requests.append(request)
        if not request.get('tools'):
            summaries.append(request)
            # Force a custom failure, then let the SDK's own complete-summary
            # handling succeed or fail. The host never repairs partial text.
            succeeds = fallback_succeeds and len(summaries)>1
            return response([], len(requests), tokens=100,
                summary=complete if succeeds else partial,
                finish='stop' if succeeds else 'length',
                completion_tokens=40 if succeeds else request['max_tokens'])
        read = [row['skill'] for kind, row in events if kind=='skill-usage'
                and row['kind']=='read' and row['stage']==Stage.TASK_CONSTRUCTION.value]
        missing = [name for name in names if name not in read]
        actions = ([('read_skill', {'name': missing[0]})] if missing else
                   [('evidence', {'sample': str(len(effects))})] if len(effects)<20 else
                   [('perform', {'proposal': 'complete after native compaction'})])
        # Underreported usage reproduces a real byte overflow before the token
        # threshold. Pi must compact, rather than a Python transcript rewrite.
        return response(actions, len(requests), tokens=1)

    def evidence(stage, args, operation, seconds):
        effects.append(args['sample'])
        return dict(reference='actual-'+args['sample'], evidence='retained evidence '*140)

    monkeypatch.setattr('coding.pi.gateway.requests.post', provider)
    actor, flow = native_actor(tmp_path, events, lambda *_: StageResult(),
        {'evidence': EvidenceTool(obj({'sample': STRING}), evidence)},
        limits=BudgetLimits(actor_seconds=25, wall_seconds=30))
    try:
        actor.step({})
        if fallback_succeeds:
            actor.step({'goal': 'recover through Pi native compaction'})
        else:
            with pytest.raises(PiCompactionError):
                actor.step({'goal': 'preserve failed native compaction'})
    finally:
        actor.close()
        (tmp_path/'native-events.json').write_text(json.dumps(events, default=str))
    attempts = [row['event'] for kind, row in events if kind=='pi-native-event'
                and row['event']['kind']=='compaction-summary']
    assert attempts and attempts[0]['successful'] is False
    assert attempts[0]['fallback']=='pi-native'
    assert effects==[str(i) for i in range(len(effects))]  # No effect replay.
    compactions = [row['event'] for kind, row in events if kind=='pi-event'
                   and row['event']['type']=='compaction_end']
    if fallback_succeeds:
        assert flow.stage==Stage.DESIGN and len(effects)==20
        checkpoints = [row['result'] for row in compactions if row.get('result')]
        assert checkpoints and all(partial not in row['summary'] for row in checkpoints)
        assert any(complete in row['summary'] for row in checkpoints)
    else:
        assert flow.pending and flow.stage==Stage.TASK_CONSTRUCTION
        assert len(flow.state['history'])==1  # Preflight only; no false commit.
        assert not any(row.get('result') for row in compactions)
    intents = [row['reservation_id'] for kind, row in events if kind=='pi-provider-request']
    ledgers = [row['reservation_id'] for kind, row in events if kind=='pi-provider']
    assert set(intents)==set(ledgers) and len(ledgers)==len(set(ledgers))
    closure = [row for kind, row in events if kind=='pi-ledger-closure'][-1]
    assert closure['provider_settled'] and not closure['accounting_unknown']
