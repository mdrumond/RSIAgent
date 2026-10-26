"""Import immutable Phase1 producers for retrospective review, never admission.

``roots`` are explicit producer roots containing state.json and waves/*/plan.json.
The returned inventory is portable once its destination artifact store is retained.
Malformed evidence is excluded; importing cannot mark a failure resolved or accept
learned memory. Contract identity omits only the local project ID.
"""
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_bytes
from benchmarks.a5kernels.phase1_curriculum import WaveSpec
from benchmarks.a5kernels.phase1_memory import Phase1ProjectMemory
from .project_verifier import ProjectReview


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    return json.loads(path.read_bytes())


def _bound(record, assignment):
    expected = dict(project_id=assignment.project_id,
        wave_fingerprint=assignment.wave_fingerprint,
        authored_ordinal=assignment.authored_ordinal,
        pre_wave_memory_sha256=assignment.pre_wave_memory_sha256)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('outcome does not match the frozen assignment')


def _artifacts(branch, store, *, strict):
    """Copy exact content once; retain logs and all readable original sources."""
    original = ContentAddressedArtifacts(branch / 'artifacts')
    artifacts, warnings = {}, []
    for path in sorted(original.objects.glob('*/*')):
        digest = path.parent.name + path.name
        try:
            store.put_bytes(original.read_bytes(digest))
        except (OSError, ValueError) as error:
            if strict:
                raise
            warnings.append(dict(artifact_sha256=digest, reason=type(error).__name__))
    journal = branch / 'evidence.jsonl'
    source = None
    if journal.exists():
        artifacts['actor-log'] = store.put_bytes(journal.read_bytes())
        try:
            ledger = EvidenceLedger(journal)
            for entry in ledger.entries:
                payload = entry.payload
                if entry.kind != 'artifact' or 'sha256' not in payload:
                    continue
                digest = payload['sha256']
                try:
                    content = original.read_bytes(digest)
                except (OSError, ValueError):
                    if strict:
                        raise
                    continue
                artifacts[f"{payload.get('role', 'artifact')}:{entry.sequence:06}"] = store.put_bytes(content)
                if payload.get('role') == 'host-observation':
                    host = json.loads(content).get('host', {})
                    if host.get('event') == 'candidate-written' and host.get('source_sha256'):
                        candidate = host['source_sha256']
                        store.put_bytes(original.read_bytes(candidate))
                        artifacts.setdefault('original-source', candidate)
                        source = candidate
        except (OSError, ValueError) as error:
            if strict:
                raise
            warnings.append(dict(reason=type(error).__name__, artifact='actor-log'))
    return artifacts, source, warnings


def _attempt(root, plan_path, plan, assignment, store):
    branch = plan_path.parent / 'branches' / assignment.project_id
    spec = assignment.spec.as_dict()
    contract = _digest(canonical_bytes({k: v for k, v in spec.items() if k != 'project_id'}))
    outcome_path, interruption_path = branch / 'outcome.json', branch / 'interruption.json'
    outcome = _read(outcome_path) if outcome_path.exists() else None
    verified = outcome is not None and outcome.get('verified') is True
    if outcome is not None:
        _bound(outcome, assignment)
    artifacts, source, warnings = _artifacts(branch, store, strict=verified)
    artifacts['contract'] = store.put_bytes(canonical_bytes(spec))
    artifacts['wave-plan'] = store.put_bytes(plan_path.read_bytes())
    memory, facts, raw = None, [], None
    if outcome is not None:
        artifacts['outcome'] = store.put_bytes(outcome_path.read_bytes())
        for filename, field, role in [('memory-delta.bin', 'memory_delta_sha256', 'memory-delta'),
                                     ('execution-evidence.bin', 'evidence_sha256', 'execution-result')]:
            data = (branch / filename).read_bytes()
            if _digest(data) != outcome.get(field):
                raise ValueError('terminal artifact does not match outcome digest')
            artifacts[role] = store.put_bytes(data)
        delta = json.loads(store.read_bytes(artifacts['memory-delta']))
        if delta.get('schema') == 'a5-phase1-adaptive-memory-delta-v1':
            if delta.get('project_id') != assignment.project_id:
                raise ValueError('memory delta project does not match assignment')
            raw = canonical_bytes(json.loads(base64.b64decode(delta['memory_base64'], validate=True)))
            memory = Phase1ProjectMemory.from_mapping(json.loads(raw))
            if memory.project_spec != assignment.spec:
                raise ValueError('memory contract differs from frozen assignment')
            source = memory.source_revision
            facts = [asdict(fact) for fact in memory.host_facts]
            original = ContentAddressedArtifacts(branch / (memory.artifact_store or 'artifacts'))
            required = {source, *memory.admitted_artifact_sha256,
                        *(fact.evidence_sha256 for fact in memory.host_facts)}
            required.update(d for lesson in memory.lessons for d in lesson.evidence_artifact_sha256)
            for digest in sorted(required):
                store.put_bytes(original.read_bytes(digest))
                artifacts.setdefault('retained:' + digest, digest)
            artifacts['historical-memory'] = store.put_bytes(raw)
        if verified:
            execution = json.loads(store.read_bytes(artifacts['execution-result']))
            if (memory is None or execution.get('classification') != 'host-verified'
                    or execution.get('source_sha256') != source
                    or outcome.get('classification') != 'host-verified'):
                raise ValueError('verified outcome lacks matching source and host evidence')
            for digest in execution.get('artifacts', {}).values():
                store.put_bytes(ContentAddressedArtifacts(branch / 'artifacts').read_bytes(digest))
                artifacts.setdefault('host-result:' + digest, digest)
    reason = 'No finalized outcome retained'
    if interruption_path.exists():
        interruption = _read(interruption_path)
        if (interruption.get('project_id') != assignment.project_id or
                interruption.get('wave_fingerprint') != assignment.wave_fingerprint):
            raise ValueError('interruption does not match frozen assignment')
        artifacts['interruption'] = store.put_bytes(interruption_path.read_bytes())
        reason = interruption.get('summary', interruption.get('classification', reason))
    if outcome is not None:
        reason = outcome.get('classification', 'unverified')
    if source is not None:
        store.read_bytes(source)
        artifacts['source'] = source
    producer = dict(root=root.name, wave_id=plan.wave_id,
        wave_fingerprint=plan.fingerprint, project_id=assignment.project_id,
        authored_ordinal=assignment.authored_ordinal)
    identity = _digest(canonical_bytes(dict(wave=plan.fingerprint, project=assignment.project_id,
        outcome=artifacts.get('outcome'), interruption=artifacts.get('interruption'),
        log=artifacts.get('actor-log'))))
    request = ProjectReview(assignment.project_id, source, verified, artifacts,
        tuple(facts), assignment.spec.hypothesis)
    row = dict(id=identity, producer=producer, spec=spec, contract_sha256=contract,
        review_request=asdict(request), reason=reason, attempts=[producer], warnings=warnings)
    return row, verified, raw, memory


def import_history(roots, store):
    """Return authenticated published memories and unresolved contract groups.

    No producer files are written. Recoveries are coalesced by complete spec minus
    project_id; hypotheses are retained because they describe distinct repair bugs.
    A verified later attempt suppresses recovery, even if it was not published.
    ``ProjectReview(**row['review_request'])`` reconstructs each review request.
    """
    memories, failures, exclusions, successful, seen = {}, {}, [], set(), set()
    attempt_count = 0
    for root in map(Path, roots):
        try:
            state = _read(root / 'state.json')
            published = {canonical_bytes(json.loads(base64.b64decode(x, validate=True)))
                         for x in state.get('memory_bytes_base64', [])}
        except (OSError, ValueError) as error:
            exclusions.append(dict(producer=root.name, reason=type(error).__name__))
            continue
        found = set()
        for plan_path in sorted(root.glob('waves/*/plan.json')):
            try:
                body = _read(plan_path)
                plan = WaveSpec.from_mapping(body['wave'])
                if body.get('wave_fingerprint') != plan.fingerprint:
                    raise ValueError('wave plan fingerprint changed')
            except (OSError, ValueError, KeyError, TypeError) as error:
                exclusions.append(dict(producer=root.name, reason=str(error)))
                continue
            for assignment in plan.assignments():
                try:
                    row, verified, raw, memory = _attempt(root, plan_path, plan, assignment, store)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    exclusions.append(dict(producer=root.name, project_id=assignment.project_id,
                                           reason=str(error)))
                    continue
                if verified and raw in published:
                    found.add(raw)
                if row['id'] in seen:
                    continue
                seen.add(row['id']); attempt_count += 1
                contract = row['contract_sha256']
                if verified:
                    successful.add(contract)
                    if raw in published:
                        found.add(raw)
                        key = _digest(raw)
                        candidate = memories.setdefault(key, dict(row, id=key, memory=memory.as_dict()))
                        if row['producer'] not in candidate['attempts']:
                            candidate['attempts'].append(row['producer'])
                    continue
                group = failures.setdefault(contract, dict(row, id=contract))
                if row['producer'] not in group['attempts']:
                    group['attempts'].append(row['producer'])
                # Keep all attempts' evidence available to recovery and review.
                if group['review_request'] != row['review_request']:
                    for role, digest in row['review_request']['artifacts'].items():
                        group['review_request']['artifacts'][row['id'] + ':' + role] = digest
        for raw in published - found:
            exclusions.append(dict(producer=root.name, memory_sha256=_digest(raw),
                                   reason='published memory has no authenticated producer'))
    pending = [row for contract, row in failures.items() if contract not in successful]
    return dict(schema='phase1-history-import-v1', memories=list(memories.values()),
        failures=pending, exclusions=exclusions,
        counts=dict(memories=len(memories), failures=len(pending), attempts=attempt_count,
                    resolved_contracts=len(set(failures) & successful), exclusions=len(exclusions)))
