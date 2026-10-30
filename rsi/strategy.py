"""Host-owned strategy evidence, persisted in the existing CAS and run log.

The host injects a configured KDB query adapter, workflow snapshot getter and
append-only log sink. Replay only that trusted log, never actor-supplied events.
Query adapters return {results: [...], negative_records: [...], provenance: {...}};
rows retain exact source content/citations. No retrieval or embedding fallback
is implemented here. Configure the adapter with the real production backend.
"""
from __future__ import annotations

from hashlib import sha256
import json
from uuid import uuid4

from rsi.contracts import Stage
from rsi.runtime import WorkflowError


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return sha256(_bytes(value)).hexdigest()


class QueryUnavailable(WorkflowError):
    """Suspend strategy work; unavailability is not a retrieval miss."""


class StrategyEvidence:
    """Single-writer evidence service; ``snapshot`` returns Workflow.state.

    ``emit`` must durably append events before returning. ``events`` is a replay
    of that same host log. CAS existence alone never grants receipt authority.
    A service belongs to one workflow identity, immutable bundle and configured
    backend. The caller owns backend construction, timeout and actor accounting.
    Backend names carry no quality certification; test_backend records the trusted
    composition's explicit declaration rather than guessing from an identifier.
    """

    def __init__(self, artifacts, emit, snapshot, bundle, query, *,
                 workflow_id, backend_identity, events=(), test_backend=False):
        if not workflow_id or not backend_identity:
            raise ValueError('workflow and configured backend identity required')
        if not callable(query):
            raise ValueError('configured KDB query callback required')
        if type(test_backend) is not bool:
            raise ValueError('test_backend must be an explicit boolean')
        self.artifacts, self.emit, self.snapshot = artifacts, emit, snapshot
        self.bundle, self.query_backend = bundle, query
        self.workflow_id, self.backend_identity = workflow_id, backend_identity
        self.test_backend = test_backend
        self.receipts = {}
        for event in events:
            if event.get('type') == 'strategy-evidence' and event.get('workflow') == workflow_id:
                self.receipts[event['reference']] = event['kind']

    def _context(self, stage=None, operation=None):
        state = self.snapshot()
        pending = state.get('pending')
        if (not pending or state.get('stopped') or state.get('outcome')
                or stage is not None and Stage(stage).value != state['stage']
                or operation is not None and operation != pending['id']):
            raise WorkflowError('no matching active workflow operation')
        return dict(workflow=self.workflow_id, operation=pending['id'],
                    stage=state['stage'], bundle=self.bundle.digest,
                    source=state['source'], history=_digest(state['history']))

    def _persist(self, kind, payload, context=None):
        record = dict(kind=kind, context=context or self._context(), payload=payload,
                      nonce=uuid4().hex)
        reference = self.artifacts.put_json(record)
        self.emit(dict(type='strategy-evidence', workflow=self.workflow_id,
                       kind=kind, reference=reference))
        self.receipts[reference] = kind
        return reference

    def _load(self, kind, reference, context):
        if self.receipts.get(reference) != kind:
            raise WorkflowError('unknown host receipt')
        record = self.artifacts.read_json(reference)
        if record['kind'] != kind or record['context'] != context:
            raise WorkflowError('receipt operation/stage/bundle/history/source mismatch')
        return record['payload']

    def record_skill_event(self, event):
        """Use as SkillUsage.emit; returns a CAS read receipt via read_receipt()."""
        context = self._context()
        if (event['operation_id'] != context['operation']
                or event['stage'] != context['stage']
                or event['bundle_hash'] != context['bundle']):
            raise WorkflowError('skill event scope mismatch')
        document = self.bundle.document(event['skill'])
        content = document.read(event['section'])
        unsigned = {k: v for k, v in event.items() if k != 'receipt_id'}
        if (event['receipt_id'] != _digest(unsigned)
                or event['skill'] not in self.bundle.skills_for(Stage(context['stage']))
                or event['skill_hash'] != document.digest
                or event['content_hash'] != sha256(content.encode()).hexdigest()):
            raise WorkflowError('invalid skill event')
        return self._persist('skill', event, context)

    def read_receipt(self, skill_event_id):
        context = self._context()
        for reference, kind in self.receipts.items():
            if kind != 'skill':
                continue
            record = self.artifacts.read_json(reference)
            event = record['payload']
            if (record['context'] == context and event['kind'] == 'read'
                    and event['receipt_id'] == skill_event_id):
                return reference
        raise WorkflowError('unknown host skill read')

    def _strategy_context(self):
        context = self._context()
        if Stage(context['stage']) not in (Stage.DESIGN, Stage.OPTIMISATION):
            raise WorkflowError('strategy evidence requires design or optimisation')
        return context

    def query(self, payload):
        context = self._strategy_context()
        # Freeze arguments before invoking a callback that may mutate its input.
        request = json.loads(_bytes(payload))
        if not isinstance(request.get('query'), str) or not request['query'].strip():
            raise ValueError('nonempty query required')
        try:
            result = self.query_backend(json.loads(_bytes(request)))
            if (not isinstance(result, dict) or not isinstance(result.get('results'), list)
                    or not isinstance(result.get('negative_records'), list)
                    or not isinstance(result.get('provenance'), dict)
                    or not result['provenance']):
                raise ValueError('query adapter must retain results, negatives and provenance')
            # None/tool-error responses are never converted to an empty hit list.
            if result.get('status', 'ok') != 'ok':
                raise ValueError('KDB did not complete query')
            result = json.loads(_bytes(result))
        except Exception as exc:
            self._persist('query-unavailable', dict(request=request,
                          error_type=type(exc).__name__), context)
            raise QueryUnavailable('KDB unavailable; strategy suspended') from exc
        return self._persist('retrieval', dict(request=request, response=result,
                             backend=self.backend_identity, test_backend=self.test_backend), context)

    def retrospect(self, *, supplement=None):
        context = self._strategy_context()
        history = json.loads(_bytes(self.snapshot()['history']))
        if _digest(history) != context['history']:
            raise WorkflowError('history changed during retrospective')
        # Retain actual outcomes, including unsuccessful work and redirections.
        failures = [i for i, item in enumerate(history)
                    if item.get('passed') is False or item.get('not_dispatched')]
        redirections = [i for i, item in enumerate(history) if item.get('redesign')]
        return self._persist('retrospective', dict(history=history, failures=failures,
                             redirections=redirections, supplement=supplement), context)

    def select(self, proposal, *, query_ref, retrospective_ref, skill_refs, decisions):
        """Log a proposal only after all prerequisites resolve.

        Each decision: {group: results|negative_records, index: int,
        action: used|rejected, reason: str, redirection?: str}. Cover every row;
        using a known negative requires explicit changed applicability/redirection.
        """
        context = self._strategy_context()
        query = self._load('retrieval', query_ref, context)
        self._load('retrospective', retrospective_ref, context)
        if not isinstance(proposal, str) or not proposal.strip() or not skill_refs:
            raise WorkflowError('proposal and skill reads required')
        self._require_skill_coverage(skill_refs, context)
        rows = query['response']
        expected = {(group, i) for group in ('results', 'negative_records')
                    for i in range(len(rows[group]))}
        seen = set()
        for decision in decisions:
            key = (decision['group'], decision['index'])
            if (key not in expected or key in seen or decision['action'] not in ('used', 'rejected')
                    or not isinstance(decision['reason'], str) or not decision['reason'].strip()):
                raise WorkflowError('each retrieved record needs a unique reasoned decision')
            if (key[0] == 'negative_records' and decision['action'] == 'used'
                    and not str(decision.get('redirection', '')).strip()):
                raise WorkflowError('negative reuse requires changed applicability/redirection')
            seen.add(key)
        if seen != expected:
            raise WorkflowError('retrieved positive and negative records must be considered')
        return self._persist('selection', dict(proposal=proposal, query_ref=query_ref,
                             retrospective_ref=retrospective_ref, skill_refs=list(skill_refs),
                             decisions=decisions), context)

    def _skill(self, reference, context):
        event = self._load('skill', reference, context)
        if event['kind'] != 'read':
            raise WorkflowError('exposure or citation is not a skill read')
        return event

    def _require_skill_coverage(self, references, context):
        reads = {self._skill(ref, context)['skill'] for ref in references}
        if not set(self.bundle.skills_for(Stage(context['stage']))) <= reads:
            raise WorkflowError('all mapped stage skills must be read')

    def _current_reads(self, context):
        return [ref for ref, kind in self.receipts.items()
                if kind == 'skill'
                and (record := self.artifacts.read_json(ref))['context'] == context
                and record['payload']['kind'] == 'read']

    def resolve(self, kind, reference, stage, operation_id):
        """Workflow.resolver-compatible; strategy completion requires selection."""
        try:
            context = self._context(stage, operation_id)
            if kind == 'skill':
                self._skill(reference, context)
                self._require_skill_coverage(self._current_reads(context), context)
                return True
            if kind not in ('retrieval', 'retrospective'):
                return False
            self._load(kind, reference, context)
            field = 'query_ref' if kind == 'retrieval' else 'retrospective_ref'
            for ref, receipt_kind in self.receipts.items():
                if receipt_kind == 'selection':
                    record = self.artifacts.read_json(ref)
                    if record['context'] == context and record['payload'][field] == reference:
                        return True
            return False
        except (WorkflowError, ValueError, KeyError, OSError):
            return False


def knowledge_db_query(database, collections, *, negative_query):
    """Adapt existing KnowledgeDB; negative_query reads the existing outcome store.

    The configured database owns embeddings. Negative lookup returns retained
    records with provenance, not heuristic text sentiment classification.
    KnowledgeAgent/SubjectKnowledge callers can provide equivalent envelopes.
    """
    from dataclasses import asdict

    def query(payload):
        results, manifests = [], {}
        for collection in collections:
            manifest = database.manifest(collection)
            manifests[collection] = json.loads(manifest.to_json())
            for hit in database.query(collection, payload['query'], limit=payload.get('limit', 5)):
                results.append(dict(collection=collection, **asdict(hit)))
        return dict(results=results, negative_records=negative_query(payload),
                    provenance=dict(collections=manifests))
    return query
