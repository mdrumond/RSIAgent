"""Typed host research contracts; accepted originals remain immutable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticProposal:
    name: str
    goal: str
    acceptance: str
    template_id: str
    parameters: dict
    parent_project_id: str | None = None


@dataclass(frozen=True)
class ResearchResult:
    proposal: DiagnosticProposal | None
    evidence_refs: tuple[str, ...]
    actor_seconds: float
    receipt_ref: str


@dataclass(frozen=True)
class ResearchQuery:
    query: str
    purpose: str


@dataclass(frozen=True)
class QueryPlan:
    rationale: str
    evidence_ids: tuple[str, ...]
    queries: tuple[ResearchQuery, ...]


@dataclass(frozen=True)
class NegativeDecision:
    query_ref: str
    index: int
    action: str
    reason: str
    redirection: str | None = None


@dataclass(frozen=True)
class DiagnosticDecision:
    rationale: str
    evidence_ids: tuple[str, ...]
    proposal: DiagnosticProposal
    negative_decisions: tuple[NegativeDecision, ...]


def parse_proposal(role, data, admitted):
    """Validate typed NOOA JSON without importing its optional framework."""
    from core.artifacts import canonical_bytes
    keys = {'rationale', 'evidence_ids', 'queries'} if role == 'research_queries' else {
        'rationale', 'evidence_ids', 'proposal', 'negative_decisions'}
    if type(data) is not dict or set(data) != keys:
        raise ValueError('invalid research proposal fields')
    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('research fields must be nonempty strings')
        return value
    rationale = text(data['rationale'])
    ids = data['evidence_ids']
    if type(ids) is not list or any(type(ref) is not str or ref not in admitted for ref in ids):
        raise ValueError('research cites unadmitted evidence')
    if role == 'research_queries':
        rows = data['queries']
        if type(rows) is not list or not rows:
            raise ValueError('documentation research requires actual queries')
        queries = []
        for row in rows:
            if type(row) is not dict or set(row) != {'query', 'purpose'}:
                raise ValueError('invalid research query')
            queries.append(ResearchQuery(text(row['query']), text(row['purpose'])))
        return QueryPlan(rationale, tuple(ids), tuple(queries))
    row = data['proposal']
    required = {'name', 'goal', 'acceptance', 'template_id', 'parameters'}
    if type(row) is not dict or not required <= row.keys() or row.keys() - required - {'parent_project_id'}:
        raise ValueError('invalid diagnostic project')
    for key in required - {'parameters'}:
        text(row[key])
    if type(row['parameters']) is not dict:
        raise ValueError('diagnostic parameters must be a JSON object')
    canonical_bytes(row['parameters'])
    if row.get('parent_project_id') is not None:
        text(row['parent_project_id'])
    proposal = DiagnosticProposal(**row)
    decisions = []
    if type(data['negative_decisions']) is not list:
        raise ValueError('negative decisions must be an array')
    for row in data['negative_decisions']:
        required = {'query_ref', 'index', 'action', 'reason'}
        if (type(row) is not dict or not required <= row.keys() or row.keys() - required - {'redirection'}
                or type(row['index']) is not int or row['index'] < 0
                or row['action'] not in ('used', 'rejected')):
            raise ValueError('invalid negative decision')
        text(row['query_ref']); text(row['reason'])
        if row['action'] == 'used':
            text(row.get('redirection'))
        decisions.append(NegativeDecision(**row))
    return DiagnosticDecision(rationale, tuple(ids), proposal, tuple(decisions))


class ResearchPending(RuntimeError):
    def __init__(self, receipt_ref, actor_seconds=None):
        super().__init__('research requires observation of the retained operation')
        self.receipt_ref, self.actor_seconds = receipt_ref, actor_seconds


class ResearchRejected(ResearchPending):
    """Known closed research; a fresh decision may use this retained feedback."""


class Researcher:
    def __init__(self, roles, query, store, *, clock=None):
        import time
        self.roles, self.query, self.store = roles, query, store
        self.clock = clock or time.monotonic

    def research(self, parent, failure, context):
        import hashlib
        import json
        import math
        from dataclasses import asdict
        from pathlib import Path
        from core.artifacts import canonical_bytes
        from rsi.python_execution import atomic_json
        from rsi.roles import AdmittedEvidence
        entered = self.clock()
        operation = context['operation_id']
        if not isinstance(operation, str) or not operation:
            raise ValueError('research operation identity required')
        for row in [failure, *(r['failure'] for r in context.get('failed_parents', [])),
                    *(r.get('provenance', {}) for r in context.get('failed_parents', []))]:
            self._validate_failure(row)
        request = dict(parent=parent.to_dict(), failure=failure,
            context={key: context[key] for key in ('templates', 'failed_parents', 'namespace',
                'original_project_id', 'memory_ref', 'skills_ref', 'configuration_ref',
                'previous_rejections', 'user_guidance', 'skill_guidance', 'kdb_guidance') if key in context})
        request_ref = self.store.put_json(request)
        path = Path(context['root']) / 'research' / (hashlib.sha256(operation.encode()).hexdigest() + '.json')
        if path.exists():
            saved = json.loads(path.read_bytes())
            if saved['request_ref'] != request_ref:
                raise ValueError('research inputs differ from retained operation')
            if saved['status'] == 'completed':
                row = self.store.read_json(saved['receipt_ref'])
                return ResearchResult(DiagnosticProposal(**row['proposal']) if row['proposal'] else None,
                    tuple(row['evidence_refs']), row['actor_seconds'], saved['receipt_ref'])
            error = ResearchRejected if saved['status'] == 'rejected' else ResearchPending
            raise error(saved['receipt_ref'], saved.get('actor_seconds') if saved['status'] == 'rejected' else None)
        deadline = context['deadline']
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('finite monotonic research deadline required')
        allowance = max(0., deadline - entered)
        wall_deadline = context.get('wall_deadline')
        state = dict(request_ref=request_ref, status='pending', actor_seconds=0., evidence_refs=[], events=[])
        state['receipt_ref'] = request_ref
        def save():
            if state['status'] == 'pending' and not state.get('settled'):
                state['receipt_ref'] = self.store.put_json(dict(schema='rsi-curriculum-research-v1',
                    status='pending', request_ref=request_ref, actor_seconds=None,
                    measured_actor_seconds=state['actor_seconds'], inflight_query=state.get('inflight_query'),
                    artifact_refs=[request_ref, *state['events'], *state['evidence_refs']]))
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, state)
        save()  # Intent precedes either provider or retrieval dispatch.
        active, finished, unknown = False, False, False
        original_observer = self.roles.observer
        def observe(event, payload):
            nonlocal active, finished, unknown
            ref = self.store.put_json(dict(event=event, payload=payload, request_ref=request_ref))
            state['events'].append(ref)
            if event == 'role_started':
                active, finished = True, False
            elif event == 'role_finished':
                elapsed = payload['elapsed_seconds']
                if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('invalid research elapsed receipt')
                state['actor_seconds'] += elapsed
                finished = True
            elif event == 'accounting_incomplete' and payload.get('start_receipt_received'):
                unknown = True
            save()
            original_observer(event, payload)
        self.roles.observer = observe
        query_refs, negatives, decision = [], set(), None
        def exhausted():
            return (state['actor_seconds'] >= allowance or
                    wall_deadline is not None and self.clock() >= wall_deadline)
        def remaining():
            if exhausted():
                raise TimeoutError('curriculum research elapsed allocation exhausted')
            return self.clock() + allowance - state['actor_seconds']
        def finish(status, proposal=None, error=None):
            seconds = None if unknown or active and not finished else state['actor_seconds']
            receipt = dict(schema='rsi-curriculum-research-v1', operation_id=operation,
                request_ref=request_ref, status=status, proposal=asdict(proposal) if proposal else None,
                evidence_refs=state['evidence_refs'], actor_seconds=seconds,
                artifact_refs=[request_ref, *state['events'], *state['evidence_refs']],
                measured_actor_seconds=state['actor_seconds'], error=error,
                decision=asdict(decision) if status == 'accepted' else None)
            state['settled'] = True
            state.update(status='completed' if status in ('accepted', 'budget_exhausted') else status,
                         receipt_ref=self.store.put_json(receipt), actor_seconds=seconds)
            save()
            return ResearchResult(proposal, tuple(state['evidence_refs']), seconds, state['receipt_ref'])
        try:
            remaining()
            guidance = {key: context[key] for key in ('user_guidance', 'configuration_ref', 'skills_ref',
                        'skill_guidance', 'kdb_guidance') if key in context}
            steering = 'Host-provided curriculum steering and frozen guidance:\n' + json.dumps(
                guidance, sort_keys=True, allow_nan=False) + '\n'
            evidence = [AdmittedEvidence(request_ref, dict(parent=parent.to_dict(),
                failure=self._facts(failure), templates=context['templates'],
                memory_ref=context.get('memory_ref'), **guidance))]
            parents = context.get('failed_parents') or [dict(project_id=parent.project_id,
                                                            spec=parent.to_dict(), failure=failure)]
            parent_refs = {}
            for row in parents:
                ref = self.store.put_json(row)
                parent_refs[row['project_id']] = ref
                evidence.append(AdmittedEvidence(ref, dict(project_id=row['project_id'],
                    spec=row['spec'], failure=self._facts(row['failure']),
                    templates=row.get('templates', context['templates']))))
            for ref in context.get('previous_rejections', []):
                evidence.append(AdmittedEvidence(ref, self._facts(self.store.read_json(ref))))
            plan = self.roles.research_queries(
                steering + 'Choose documentation/KDB queries to investigate the admitted failures before proposing a diagnostic.',
                evidence, deadline=remaining(), wall_deadline=wall_deadline)
            if not isinstance(plan, QueryPlan) or not plan.queries:
                raise ValueError('a diagnostic requires prior model-selected documentation research')
            for query in plan.queries:
                remaining()
                state['inflight_query'] = dict(query=asdict(query), deadline=remaining())
                observe('research_query_started', state['inflight_query'])
                started = self.clock()
                try:
                    cutoff = remaining()
                    if wall_deadline is not None:
                        cutoff = min(cutoff, wall_deadline)
                    response = self.query({'query': query.query}, deadline=cutoff)
                finally:
                    elapsed = max(0., self.clock() - started)
                    state['actor_seconds'] += elapsed
                    state.pop('inflight_query', None)
                    observe('research_query_finished', dict(query=asdict(query), elapsed_seconds=elapsed))
                if (not isinstance(response, dict) or not isinstance(response.get('provenance'), dict)
                        or not response['provenance'] or any(not isinstance(response.get(k), list)
                        for k in ('results', 'negative_records'))):
                    raise ValueError('research retrieval requires actual rows and provenance')
                ref = self.store.put_json(dict(schema='rsi-curriculum-retrieval-v1',
                    request_ref=request_ref, query=asdict(query), response=response))
                query_refs.append(ref); state['evidence_refs'].append(ref); save()
                shown = {group: [dict(index=i, artifact_ref=self.store.put_json(row), facts=self._facts(row))
                    for i, row in enumerate(response[group])] for group in ('results', 'negative_records')}
                negatives.update((ref, i) for i in range(len(response['negative_records'])))
                evidence.append(AdmittedEvidence(ref, dict(query=asdict(query), **shown,
                                                          provenance=response['provenance'])))
            remaining()
            decision = self.roles.diagnostic(
                steering + 'Use the actual retrieved documents and negatives to choose a simpler host-advertised diagnostic. '
                'Cite every query receipt and the selected parent. Address every negative by query_ref/index; '
                'reuse needs a changed-applicability redirection. Preserve original cases and outcomes.',
                evidence, deadline=remaining(), wall_deadline=wall_deadline)
            remaining()
            if not isinstance(decision, DiagnosticDecision):
                raise ValueError('typed diagnostic decision required')
            proposal = decision.proposal
            selected = proposal.parent_project_id or parent.project_id
            if selected not in parent_refs or not {*query_refs, parent_refs[selected]} <= set(decision.evidence_ids):
                raise ValueError('diagnostic must cite retrieved research and its admitted parent')
            selected_row = next(row for row in parents if row['project_id'] == selected)
            self._validate_template(proposal, selected, selected_row.get('templates', context['templates']))
            seen = set()
            for row in decision.negative_decisions:
                key = (row.query_ref, row.index)
                if (key not in negatives or key in seen or not row.reason.strip()
                        or row.action not in ('used', 'rejected')
                        or row.action == 'used' and not (row.redirection or '').strip()):
                    raise ValueError('every negative needs a decision; reuse needs redirection')
                seen.add(key)
            if seen != negatives:
                raise ValueError('diagnostic omitted a retrieved negative')
            return finish('accepted', proposal)
        except Exception as exc:
            if unknown or active and not finished:
                result = finish('pending', error=type(exc).__name__)
                raise ResearchPending(result.receipt_ref, result.actor_seconds) from exc
            if isinstance(exc, TimeoutError) and exhausted():
                return finish('budget_exhausted')
            result = finish('rejected', error=str(exc))
            raise ResearchRejected(result.receipt_ref, result.actor_seconds) from exc
        finally:
            self.roles.observer = original_observer

    def _facts(self, row):
        """Useful bounded facts; complete documents and diagnostics stay in CAS."""
        if not isinstance(row, dict):
            return str(row)[:2048]
        result = {k: v[:2048] if isinstance(v, str) else v for k, v in row.items()
                  if k in ('text', 'path', 'repository', 'revision', 'collection', 'goal', 'stage',
                           'status', 'outcome', 'reason', 'summary', 'error', 'project_id',
                           'classification', 'host_outcome', 'source_hash', 'memory_id', 'kind')
                  and isinstance(v, (str, int, float, bool))}
        memory = row.get('memory', row)
        if isinstance(memory, dict) and isinstance(memory.get('proposal'), dict):
            result['memory'] = self._facts(memory['proposal'])
        if isinstance(row.get('proof_scope'), dict):
            result['proof_scope'] = self._facts(row['proof_scope'])
        observed = row.get('observations')
        observation_ref = row.get('observations_ref')
        if observation_ref:
            retained = self.store.read_json(observation_ref)
            if (not isinstance(retained, dict) or retained.get('schema') != 'rsi-project-observations-v1'
                    or observed is not None and observed != retained):
                raise ValueError('admitted observation reference differs from host facts')
            observed = retained
        if isinstance(observed, dict):
            observation_ref = observation_ref or self.store.put_json(observed)
            commands = observed.get('commands', {}).get('items', [])
            shown = []
            for command in commands[-3:]:
                fact = {k: command[k] for k in ('command', 'classification', 'returncode', 'numeric',
                    'source_hash', 'final_source_scope') if k in command}
                fact['artifact_ref'] = self.store.put_json(command)
                for stream in ('stdout', 'stderr'):
                    preview = command.get(stream, {})
                    preview = preview if isinstance(preview, dict) else {'text': str(preview)}
                    text = preview.get('text', '')
                    fact[stream] = dict(text=text[:1024], truncated=len(text) > 1024 or preview.get('truncated', False))
                    if isinstance(preview, dict) and preview.get('reference'):
                        fact[stream]['reference'] = preview['reference']
                    if stream == 'stdout' and len(text.encode()) <= 2048:
                        import json
                        try:
                            value = json.loads(text, parse_constant=lambda value: value)
                        except ValueError:
                            pass
                        else:
                            if isinstance(value, (dict, list)):
                                fact['structured_stdout'] = value
                shown.append(fact)
            result['observations'] = dict(artifact_ref=observation_ref, outcome=observed.get('outcome'),
                applicability=observed.get('applicability'), limitations=observed.get('limitations'),
                validation=observed.get('validation'), commands=dict(items=shown,
                    total=observed.get('commands', {}).get('total', len(commands)),
                    truncated=len(commands) > len(shown) or observed.get('commands', {}).get('truncated', False)))
        return result

    def actor_context(self, receipt_ref):
        """Give the canonical child useful research facts, not only opaque hashes."""
        import json
        from core.artifacts import canonical_bytes
        receipt = self.store.read_json(receipt_ref)
        if receipt.get('status') != 'accepted':
            raise ValueError('actor research context requires an accepted decision')
        decision = receipt.get('decision') or {}
        if not decision:
            for ref in receipt['artifact_refs']:
                event = self.store.read_json(ref)
                payload = event.get('payload', {})
                if event.get('event') == 'role_proposal' and payload.get('role') == 'diagnostic':
                    decision = payload['proposal']
        request = self.store.read_json(receipt['request_ref'])
        selected = receipt['proposal'].get('parent_project_id') or request['parent']['project_id']
        failure = next((r['failure'] for r in request['context'].get('failed_parents', [])
                        if r['project_id'] == selected), request['failure'])
        facts = self._facts(failure)
        commands = facts.get('observations', {}).get('commands', {}).get('items', [])
        queries, documents = [], []
        for ref in receipt['evidence_refs'][:2]:
            query = self.store.read_json(ref)
            queries.append(dict(reference=ref, query=query['query']['query']))
            for row in query['response']['results'][:1]:
                documents.append(dict(artifact_ref=self.store.put_json(row), facts=self._facts(row)))
        negatives = decision.get('negative_decisions', [])
        value = dict(receipt_ref=receipt_ref, rationale=decision.get('rationale', ''),
            template_id=receipt['proposal']['template_id'], parent_project_id=selected,
            host_failure={k: facts[k] for k in ('outcome', 'classification', 'error', 'source_hash') if k in facts},
            commands=commands[-2:], queries=queries, documents=documents[:1],
            negative_decisions=negatives[:1], negative_decision_count=len(negatives),
            query_count=len(receipt['evidence_refs']),
            presentation='Bounded excerpts; complete role/query/command records are retained in CAS.')
        # Do not repeat unrelated observations or source bodies from document hits.
        for doc in value['documents']:
            doc['facts'] = {k: v for k, v in doc['facts'].items()
                            if k in ('text', 'path', 'repository', 'revision', 'collection', 'summary', 'reason')}
        identities = {'receipt_ref', 'reference', 'artifact_ref', 'query_ref', 'source_hash',
                      'template_id', 'parent_project_id', 'path'}
        def bounded(item, size, key=None):
            if isinstance(item, str) and key not in identities:
                return item.encode()[:size].decode(errors='ignore')
            if isinstance(item, dict):
                return {k: bounded(v, size, k) for k, v in item.items()}
            if isinstance(item, list):
                return [bounded(v, size) for v in item]
            return item
        for size in (240, 160, 96, 48):
            shown = bounded(value, size)
            if len(canonical_bytes(shown)) <= 2500:
                return shown
        if len(canonical_bytes(shown)) <= 4096:
            return shown
        raise ValueError('mandatory research context identities exceed the actor handoff budget')

    def _validate_failure(self, row):
        """Historical service envelopes remain archival, never new payload roots."""
        if not isinstance(row, dict):
            raise ValueError('admitted failure facts must be an object')
        def service_envelope(value):
            return (isinstance(value, dict) and isinstance(value.get('result'), dict)
                    and isinstance(value['result'].get('events'), list))
        if service_envelope(row) or isinstance(row.get('events'), list):
            raise ValueError('pass admitted failure facts; keep historical events in origin-task-archive')
        if row.get('artifact_ref') and service_envelope(self.store.read_json(row['artifact_ref'])):
            raise ValueError('historical service-result ID must use an archival identity field')

    @staticmethod
    def _validate_template(proposal, parent_id, templates):
        from core.artifacts import canonical_bytes
        if not isinstance(proposal, DiagnosticProposal):
            raise ValueError('typed diagnostic project required')
        canonical_bytes(proposal.parameters)
        candidates = [r for r in templates if r['template_id'] == proposal.template_id
                      and r.get('parent_project_id', parent_id) == parent_id]
        if len(candidates) != 1:
            raise ValueError('diagnostic must use an advertised template for its selected parent')
        schema = candidates[0]['parameters']
        if type(proposal.parameters) is not dict or set(proposal.parameters) != set(schema):
            raise ValueError('diagnostic parameters differ from host template')
        for key, rule in schema.items():
            values = proposal.parameters[key]
            if (rule['type'] != 'array' or type(values) is not list
                    or not rule['minItems'] <= len(values) <= rule['maxItems']
                    or any(type(v) is not int or v not in rule['items']['enum'] for v in values)
                    or rule.get('uniqueItems') and len(set(values)) != len(values)):
                raise ValueError('diagnostic parameters exceed advertised parent cases')
