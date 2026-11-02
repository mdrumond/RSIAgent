"""Typed host proposals over the same native Pi sessions used by coding actors."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Mapping
from uuid import uuid4

from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from rsi.python_execution import atomic_json


class RoleFailure(RuntimeError):
    """A known-finished role failed; its elapsed receipt has already been retained."""

    def __init__(self, role, error_type, message):
        self.role, self.error_type, self.message = role, error_type, message
        super().__init__(f'{role} {error_type}: {message}')


def _error_details(exc, secret=None):
    message = str(exc)
    if secret:
        message = message.replace(secret, '[redacted]')
    return dict(type=type(exc).__name__.encode()[:128].decode(errors='ignore'),
                message=message.encode()[:2048].decode(errors='ignore'))


@dataclass(frozen=True)
class AdmittedEvidence:
    evidence_id: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class ProjectIdea:
    name: str
    goal: str
    acceptance: str


@dataclass(frozen=True)
class SkillEdit:
    skill: str
    operation: Literal['replace', 'remove', 'append']
    content: str


@dataclass(frozen=True)
class ReviewFinding:
    severity: Literal['P0', 'P1', 'P2', 'P3']
    location: str
    issue: str


@dataclass(frozen=True)
class CurriculumProposal:
    rationale: str
    evidence_ids: tuple[str, ...]
    projects: tuple[ProjectIdea, ...]


@dataclass(frozen=True)
class SkillProposal:
    rationale: str
    evidence_ids: tuple[str, ...]
    edits: tuple[SkillEdit, ...]


@dataclass(frozen=True)
class ReviewProposal:
    rationale: str
    evidence_ids: tuple[str, ...]
    findings: tuple[ReviewFinding, ...]


from rsi.curriculum_research import DiagnosticDecision, DiagnosticProposal, QueryPlan, ResearchQuery


_PROPOSALS = {'research_queries': (QueryPlan, 'queries', ResearchQuery),
              'diagnostic': (DiagnosticDecision, 'proposal', DiagnosticProposal),
              'curriculum': (CurriculumProposal, 'projects', ProjectIdea),
              'skillwriter': (SkillProposal, 'edits', SkillEdit),
              'review': (ReviewProposal, 'findings', ReviewFinding)}


def _proposal(role, data, admitted):
    """Validate domain proposals before the host applies any state change."""
    if role == 'investigation':
        from rsi.curriculum_research import InvestigationIdea
        if (type(data) is not dict or set(data) != {'question', 'goal', 'why'}
                or any(type(v) is not str or not v.strip() for v in data.values())):
            raise ValueError('investigation requires question, goal and why')
        return InvestigationIdea(**data)
    if role in ('research_queries', 'diagnostic'):
        from rsi.curriculum_research import parse_proposal
        return parse_proposal(role, data, admitted)
    outer, name, inner = _PROPOSALS[role]
    if type(data) is not dict or set(data) != {'rationale', 'evidence_ids', name}:
        raise ValueError('invalid proposal fields')
    ids, rows = data['evidence_ids'], data[name]
    if not isinstance(data['rationale'], str) or not data['rationale'].strip():
        raise ValueError('proposal requires rationale')
    if type(ids) is not list or any(type(x) is not str or x not in admitted for x in ids):
        raise ValueError('proposal cites unadmitted evidence')
    if type(rows) is not list or (role == 'curriculum' and not rows):
        raise ValueError('invalid proposal entries')
    entries = []
    for row in rows:
        if type(row) is not dict or set(row) != {f.name for f in fields(inner)}:
            raise ValueError('invalid proposal entry fields')
        if any(type(v) is not str or (not v.strip() and k != 'content') for k, v in row.items()):
            raise ValueError('invalid proposal entry value')
        if role == 'skillwriter' and row['operation'] not in ('replace', 'remove', 'append'):
            raise ValueError('invalid skill operation')
        if role == 'review' and row['severity'] not in ('P0', 'P1', 'P2', 'P3'):
            raise ValueError('invalid finding severity')
        entries.append(inner(**row))
    return outer(data['rationale'], tuple(ids), tuple(entries))


def proposal_schema(role):
    """Native tool schemas; host validation remains authoritative."""
    text = dict(type='string', minLength=1, pattern=r'\S')
    def obj(properties, optional=()):
        return dict(type='object', properties=properties,
                    required=[k for k in properties if k not in optional], additionalProperties=False)
    def array(items, **limits):
        return dict(type='array', items=items, **limits)
    if role == 'investigation':
        return obj(dict(question=text, goal=text, why=text))
    if role == 'research_queries':
        payload = dict(queries=array(obj(dict(query=text, purpose=text)), minItems=1))
    elif role == 'diagnostic':
        proposal = obj(dict(name=text, goal=text, acceptance=text, template_id=text,
            parameters=dict(type='object'), parent_project_id=dict(anyOf=[text, dict(type='null')])),
            optional=('parent_project_id',))
        negative = dict(query_ref=text, index=dict(type='integer', minimum=0), reason=text)
        payload = dict(proposal=proposal, negative_decisions=array(dict(anyOf=[
            obj(dict(negative, action=dict(const='used'), redirection=text)),
            obj(dict(negative, action=dict(const='rejected'), redirection=dict(anyOf=[text, dict(type='null')])),
                optional=('redirection',))])))
    else:
        _, name, _ = _PROPOSALS[role]
        properties = {'curriculum': dict(name=text, goal=text, acceptance=text),
            'skillwriter': dict(skill=text, operation=dict(enum=['replace', 'remove', 'append']),
                               content=dict(type='string')),
            'review': dict(severity=dict(enum=['P0', 'P1', 'P2', 'P3']), location=text, issue=text)}[role]
        payload = {name: array(obj(properties), **(dict(minItems=1) if role == 'curriculum' else {}))}
    return obj(dict(rationale=text, evidence_ids=array(text), **payload))


class NativeRoles:
    """One durable SDK session per host role operation; evidence is read-only.

    Pi owns tools, provider calls and continuation. Proposals never execute
    projects or apply skills. Retained unknown operations cannot be replayed.
    """
    def __init__(self, root, profile, *, observer, environment, store=None,
                 operation_id=None, session_factory=None, files_factory=None, node_modules=None,
                 setup_seconds=120., receipt_grace_seconds=None, clock=time.monotonic):
        from coding.pi.native import NativeSession
        self.root, self.profile = Path(root), profile
        self.observer, self.environment, self.clock = observer, dict(environment), clock
        self.store = store or ContentAddressedArtifacts(self.root/'artifacts')
        self.operation_id = operation_id or (lambda: uuid4().hex)
        self.session_factory = session_factory or NativeSession
        self.files_factory, self.node_modules = files_factory, node_modules
        if type(setup_seconds) not in (int, float) or not math.isfinite(setup_seconds) or setup_seconds <= 0:
            raise ValueError('native role setup_seconds must be finite and positive')
        self.setup_seconds = setup_seconds

    def curriculum(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('curriculum', prompt, evidence, deadline, wall_deadline)

    def investigation(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('investigation', prompt, evidence, deadline, wall_deadline)

    def skillwriter(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('skillwriter', prompt, evidence, deadline, wall_deadline)

    def review(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('review', prompt, evidence, deadline, wall_deadline)

    def research_queries(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('research_queries', prompt, evidence, deadline, wall_deadline)

    def diagnostic(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('diagnostic', prompt, evidence, deadline, wall_deadline)

    def _run(self, role, prompt, evidence, deadline, wall_deadline):
        from coding.pi.native import NativeError, NativeTool
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('finite native role deadline required')
        allowance = deadline-self.clock()
        if allowance <= 0:
            raise TimeoutError('native role actor allocation exhausted')
        if wall_deadline is not None and (type(wall_deadline) not in (int, float)
                or not math.isfinite(wall_deadline) or wall_deadline <= self.clock()):
            raise TimeoutError('native role wall allocation exhausted')
        evidence = list(evidence)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('role prompt must be nonempty')
        if any(not isinstance(e, AdmittedEvidence) or not isinstance(e.evidence_id, str)
               or not e.evidence_id for e in evidence) or len({e.evidence_id for e in evidence}) != len(evidence):
            raise ValueError('admitted evidence requires unique nonempty IDs')
        raw = canonical_bytes(dict(role=role, prompt=prompt, evidence=[asdict(e) for e in evidence]))
        secret = self.environment.get(self.profile.identity.credential_env)
        if secret and secret in raw.decode():
            raise ValueError('credential cannot appear in role inputs')
        request_ref = self.store.put_bytes(raw)
        operation = self.operation_id()
        if not isinstance(operation, str) or not operation:
            raise ValueError('native role requires a host operation identity')
        root = self.root/hashlib.sha256((operation+':'+role).encode()).hexdigest()
        path = root/'role.json'
        if path.exists():
            saved = json.loads(path.read_bytes())
            if saved['request_ref'] != request_ref:
                raise ValueError('retained native role inputs differ')
            if saved['status'] == 'completed':
                return _proposal(role, self.store.read_json(saved['proposal_ref']), {e.evidence_id for e in evidence})
            if saved['status'] == 'failed':
                raise RoleFailure(role, saved['error']['type'], saved['error']['message'])
            if saved['status'] == 'preparing':
                raise RoleFailure(role, 'SetupIncomplete', 'Host setup interrupted before native prompt')
            self.observer('accounting_incomplete', dict(role=role, start_receipt_received=True,
                native_root=str(root), request_ref=request_ref))
            raise NativeError('retained native role requires observation; never replay')
        root.mkdir(parents=True)
        state = dict(operation_id=operation, request_ref=request_ref, status='preparing')
        atomic_json(path, state)
        inputs = root/'evidence'
        inputs.mkdir()
        prompt_file = inputs/'instructions.txt'
        prompt_file.write_text(prompt)
        admitted_files = [prompt_file]
        manifest = []
        for item in evidence:
            content = json.dumps(dict(evidence_id=item.evidence_id, data=item.data),
                                 indent=2, ensure_ascii=False, allow_nan=False).encode()
            reference = self.store.put_bytes(content)
            target = inputs/(reference+'.json')
            target.write_bytes(content)
            admitted_files.append(target)
            manifest.append(dict(evidence_id=item.evidence_id, path=str(target), bytes=len(content), reference=reference))
        manifest_file = inputs/'manifest.json'
        manifest_ref = self.store.put_json(manifest)
        manifest_file.write_bytes(canonical_bytes(manifest))
        admitted_files.append(manifest_file)
        proposal = None
        def submit(data, context):
            nonlocal proposal
            if context['remaining']() <= 0 or context['operation_id'] != operation:
                raise ValueError('native proposal operation or elapsed allocation differs')
            if proposal is not None:
                raise ValueError('role proposal already committed')
            value = _proposal(role, data, {e.evidence_id for e in evidence})
            state['proposal_ref'] = self.store.put_json(data)
            atomic_json(path, state)
            proposal = value
            self.observer('role_proposal', dict(role=role, proposal_ref=state['proposal_ref'],
                tool_call_id=context['tool_call_id']))
            return dict(content=[dict(type='text', text='Proposal retained for host validation.')], terminate=True)
        if self.files_factory is None:
            from coding.pi.filesystem import NativeFiles
            files_factory = NativeFiles
        else:
            files_factory = self.files_factory
        resources = files_factory(root, store=self.store, read_paths=admitted_files,
            record=lambda kind, payload: self.observer('role_native_file',
                dict(role=role, kind=kind, **payload)))
        def observe(event):
            reference = self.store.put_json(event)
            self.observer('role_native_event', dict(role=role, type=event['type'], artifact_ref=reference))
        session = self.session_factory(root/'native', cwd=root,
            profile_id=self.profile.identity.profile_id, environ=self.environment, node_modules=self.node_modules,
            tools=[NativeTool('submit_proposal', 'Submit the completed typed role proposal.',
                              proposal_schema(role), submit)], observer=observe,
            system_prompt='Read admitted instructions and evidence using native read. '
                          'Perform the noncoding role and commit with submit_proposal. '
                          'Only the host applies proposals. Project execution and edits are unavailable.',
            **resources.session_options())
        started = failure = settlement = None
        try:
            timeout = self.setup_seconds
            if wall_deadline is not None:
                timeout = min(timeout, max(0., wall_deadline-self.clock()))
            session.prepare(timeout=timeout)
            if wall_deadline is not None and self.clock() >= wall_deadline:
                raise TimeoutError('native role wall allocation exhausted during setup')
            started = self.clock()
            cutoff = started+allowance
            if wall_deadline is not None:
                cutoff = min(cutoff, wall_deadline)
            state.update(status='pending')
            atomic_json(path, state)
            self.observer('role_started', dict(role=role, request_ref=request_ref,
                evidence_manifest_ref=manifest_ref, native_root=str(root/'native')))
            settlement = session.prompt('Read '+str(prompt_file)+' and '+str(manifest_file)+
                '; use the admitted evidence files, then submit the proposal.',
                operation_id=operation, remaining=lambda: max(0., cutoff-self.clock()))
            if self.clock() >= cutoff:
                raise TimeoutError('native role actor allocation exhausted')
            if not settlement.provider_settled or settlement.pending_tools:
                raise NativeError('native role closure unknown', settlement=settlement)
            if proposal is None:
                raise RoleFailure(role, 'Blocked', 'Native role settled without a committed proposal')
            return proposal
        except BaseException as error:
            failure = error
            settlement = getattr(error, 'settlement', None) or settlement
            raise
        finally:
            elapsed = max(0., self.clock()-started) if started is not None else None
            try:
                session.close()
            finally:
                if settlement is None and isinstance(failure, KeyboardInterrupt) and session.provider_settled:
                    from coding.pi.native import NativeSettlement
                    settlement = NativeSettlement('cancelled', True, tuple(session.pending_tools),
                        session.session_file, tuple(session.usage))
                known = settlement is not None and settlement.provider_settled and not settlement.pending_tools
                if started is None:
                    state.update(status='failed', error=_error_details(failure or RuntimeError('role setup failed'), secret))
                elif known:
                    state.update(status='completed' if failure is None else 'failed', actor_seconds=elapsed)
                    if failure is not None:
                        state['error'] = _error_details(failure, secret)
                    self.observer('role_finished', dict(role=role, success=failure is None,
                        elapsed_seconds=elapsed, **({'error': state['error']} if failure is not None else {})))
                else:
                    state.update(status='pending', actor_seconds=None)
                    self.observer('accounting_incomplete', dict(role=role, start_receipt_received=True,
                        native_root=str(root/'native'), request_ref=request_ref))
                atomic_json(path, state)
