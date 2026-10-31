"""Typed, proposal-only NOOA roles over the host's registered direct provider.

Importing this module needs only the standard library. Install ``rsi[roles]``
with Python 3.12/3.13 to invoke a role. The host supplies admitted evidence,
absolute monotonic deadlines, and an observer for artifacts/accounting. Calls
are synchronous, including the bridge's async entry, so cancellation cannot
leave an unobserved provider thread running. Provider transport enforces timeout.
No tools, execution, memory storage, or proposal application is delegated here.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import tempfile
import subprocess
import sys
import time
from typing import Any, Literal, Mapping


def _wall_remaining(clock, deadline):
    if deadline is None:
        return None
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('wall_deadline must be finite monotonic seconds or None')
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError('noncoding role wall deadline exhausted')
    return remaining


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


@lru_cache(maxsize=1)
def _framework():
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError('NOOA roles require Python 3.12 or 3.13; RSI core supports 3.10+')
    try:
        from nooa import Agent, strategy
        from nooa.config.strategy_config import PredictConfig
        from nooa.strategies.predict import PredictStrategy
        from nooa.unifiedllm.unifiedllm import UnifiedLLM, LLMResponse
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError('Install rsi[roles] to use typed noncoding roles') from exc

    class Proposal(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
        rationale: str = Field(min_length=1)
        evidence_ids: list[str]

    class Project(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        name: str = Field(min_length=1)
        goal: str = Field(min_length=1)
        acceptance: str = Field(min_length=1)

    class ResearchQueryModel(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        query: str = Field(min_length=1)
        purpose: str = Field(min_length=1)

    class ResearchQueries(Proposal):
        queries: list[ResearchQueryModel] = Field(min_length=1)

    class DiagnosticProject(Project):
        template_id: str = Field(min_length=1)
        parameters: dict[str, Any]
        parent_project_id: str | None = None

    class NegativeDecisionModel(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        query_ref: str
        index: int
        action: Literal['used', 'rejected']
        reason: str = Field(min_length=1)
        redirection: str | None = None

    class Diagnostic(Proposal):
        proposal: DiagnosticProject
        negative_decisions: list[NegativeDecisionModel]

    class Edit(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        skill: str = Field(min_length=1)
        operation: Literal['replace', 'remove', 'append']
        content: str

    class Finding(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        severity: Literal['P0', 'P1', 'P2', 'P3']
        location: str = Field(min_length=1)
        issue: str = Field(min_length=1)

    class Curriculum(Proposal):
        projects: list[Project] = Field(min_length=1)

    class Skills(Proposal):
        edits: list[Edit]

    class Review(Proposal):
        findings: list[Finding]

    class Roles(Agent):
        @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
        async def curriculum(self, role_prompt: str, admitted_evidence: str) -> Curriculum:
            """Propose a curriculum from the role prompt and admitted evidence. Cite evidence IDs."""
            ...

        @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
        async def research_queries(self, role_prompt: str, admitted_evidence: str) -> ResearchQueries:
            """Choose documentation/KDB queries before proposing a diagnostic project."""
            ...

        @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
        async def diagnostic(self, role_prompt: str, admitted_evidence: str) -> Diagnostic:
            """Use actual retrieved facts and address every negative before choosing a host template."""
            ...

        @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
        async def skillwriter(self, role_prompt: str, admitted_evidence: str) -> Skills:
            """Propose evidence-supported skill edits. Prioritise error prevention and recovery."""
            ...

        @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
        async def review(self, role_prompt: str, admitted_evidence: str) -> Review:
            """Review admitted evidence; report actionable findings, or an empty findings list."""
            ...

    class RegisteredClient(UnifiedLLM):
        def __init__(self, owner, deadline):
            super().__init__(model=owner.profile.model)
            self.owner, self.deadline = owner, deadline

        @property
        def context_window(self):
            # The registered provider enforces its authoritative byte budget.
            return None

        def call(self, messages, tools=None, output_model=None, **kwargs):
            if tools:
                raise ValueError('noncoding roles cannot dispatch tools')
            normalised = []
            for message in messages:
                content = message.get('content')
                if isinstance(content, list):
                    if any(block.get('type') != 'text' for block in content):
                        raise ValueError('noncoding roles accept text evidence only')
                    content = '\n'.join(block['text'] for block in content)
                if not isinstance(content, str) or message['role'] not in ('system', 'user', 'assistant'):
                    raise ValueError('unsupported NOOA message')
                normalised.append({'role': message['role'], 'content': content})
            normalised.append({'role': 'user', 'content': 'Return JSON matching this schema:\n' +
                               json.dumps(output_model.model_json_schema(), sort_keys=True)})
            result = self.owner.provider.complete(
                self.owner.profile, normalised, attempt_observer=self.owner.observer,
                max_seconds=self.owner._remaining(self.deadline))
            self.owner._remaining(self.deadline)
            self.owner.observer('role_response', {'text': result.text,
                                                 'completion_tokens': result.completion_tokens})
            return LLMResponse(raw_response=None, content=result.text, tool_calls=[],
                               finish_reason='stop', assistant_message={'role': 'assistant',
                                                                       'content': result.text},
                               usage={'completion_tokens': result.completion_tokens})

        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            return self.call(messages, tools, output_model, **kwargs)

    return Roles, RegisteredClient


class NoncodingRoles:
    """One fresh NOOA agent per proposal; no hidden retries or cross-task history.

    ``observer(event, payload)`` receives role lifecycle, provider attempts, raw
    response and validated proposal events. Persist these in host-owned artifacts.
    Setup is outside the measured role interval; role work is charged even on
    failure. Calls must not overlap on a single host controller.
    """

    def __init__(self, provider, profile, *, observer, clock=time.monotonic):
        from benchmarks.a5kernels.phase1_provider import direct_profile
        if profile != direct_profile(profile.identity):
            raise ValueError('role profile must use a registered provider identity')
        self.provider, self.profile = provider, profile
        self.observer, self.clock = observer, clock

    def _remaining(self, deadline):
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('noncoding role actor time exhausted')
        return remaining

    def curriculum(self, role_prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('curriculum', role_prompt, evidence, deadline, wall_deadline)

    def research_queries(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('research_queries', prompt, evidence, deadline, wall_deadline)

    def diagnostic(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('diagnostic', prompt, evidence, deadline, wall_deadline)

    def skillwriter(self, role_prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('skillwriter', role_prompt, evidence, deadline, wall_deadline)

    def review(self, role_prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('review', role_prompt, evidence, deadline, wall_deadline)

    def _run(self, role, prompt, evidence, deadline, wall_deadline=None):
        _wall_remaining(self.clock, wall_deadline)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('role prompt must be nonempty')
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('deadline must be finite monotonic seconds')
        ids, rows = set(), []
        for item in evidence:
            if not isinstance(item, AdmittedEvidence) or not item.evidence_id or item.evidence_id in ids:
                raise ValueError('admitted evidence requires unique nonempty IDs')
            ids.add(item.evidence_id)
            rows.append({'evidence_id': item.evidence_id, 'data': dict(item.data)})
        encoded = json.dumps(rows, allow_nan=False, sort_keys=True)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('call synchronous roles outside an active asyncio loop')
        allowance = self._remaining(deadline)
        Roles, Client = _framework()
        client = Client(self, deadline)
        agent = Roles(llm=client)
        started = self.clock()
        _wall_remaining(self.clock, wall_deadline)
        deadline = client.deadline = min(started + allowance, wall_deadline) if wall_deadline is not None else started + allowance
        success, error = False, None
        self.observer('role_started', {'role': role, 'provider_identity': self.profile.identity.as_dict(),
                                       'prompt': prompt, 'evidence': rows})
        try:
            result = asyncio.run(getattr(agent, role)(role_prompt=prompt, admitted_evidence=encoded))
            self._remaining(deadline)
            proposal = _proposal(role, result.model_dump(mode='json'), ids)
            self.observer('role_proposal', {'role': role, 'proposal': result.model_dump(mode='json')})
            success = True
            return proposal
        except Exception as exc:
            secret = getattr(self.provider, '_credentials', {}).get(self.profile.identity.credential_env)
            error = _error_details(exc, secret)
            raise
        finally:
            receipt = dict(role=role, success=success, elapsed_seconds=self.clock() - started)
            if error is not None:
                receipt['error'] = error
            self.observer('role_finished', receipt)


from rsi.curriculum_research import DiagnosticDecision, DiagnosticProposal, QueryPlan, ResearchQuery


_PROPOSALS = {'research_queries': (QueryPlan, 'queries', ResearchQuery),
              'diagnostic': (DiagnosticDecision, 'proposal', DiagnosticProposal),
              'curriculum': (CurriculumProposal, 'projects', ProjectIdea),
              'skillwriter': (SkillProposal, 'edits', SkillEdit),
              'review': (ReviewProposal, 'findings', ReviewFinding)}


def _proposal(role, data, admitted):
    """Validate the same typed wire result without loading NOOA in the host."""
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


_MACHINERY_PACKAGES = ('rsi', 'core', 'coding', 'config', 'env', 'llm', 'optimizer',
                       'explore', 'benchmarks', 'dashboard', 'integrations', 'tools')


@contextmanager
def _machinery_projection(check=lambda: None):
    """Copy only this revision's owned machinery, never host site-packages.

    A wheel's distribution inventory is authoritative. Source and copied-instance
    layouts use their explicit package directories, with dependency/cache trees
    excluded. Cleanup follows worker reaping, including failures during spawn.
    """
    from importlib.metadata import distribution, PackageNotFoundError
    machinery = Path(__file__).resolve().parent.parent
    check()
    with tempfile.TemporaryDirectory(prefix='rsi-roles-') as temporary:
        root = Path(temporary)
        def copy(source, target):
            check()
            with open(source, 'rb') as src, open(target, 'wb') as dst:
                while chunk := src.read(65536):
                    check()
                    dst.write(chunk)
            shutil.copystat(source, target)
            check()
            return target
        if (machinery / 'pyproject.toml').is_file():
            for name in _MACHINERY_PACKAGES:
                source = machinery / name
                if source.is_dir():
                    shutil.copytree(source, root / name, copy_function=copy, ignore=shutil.ignore_patterns(
                        '__pycache__', '*.pyc', '.git', 'node_modules', '.venv', 'build', 'dist'))
        else:
            try:
                installed = distribution('rsi')
            except PackageNotFoundError as exc:
                raise RuntimeError('cannot identify installed RSI machinery') from exc
            if Path(installed.locate_file('rsi/roles.py')).resolve() != Path(__file__).resolve():
                raise RuntimeError('RSI distribution does not match controller machinery')
            for entry in installed.files or ():
                if entry.parts[0] not in _MACHINERY_PACKAGES or '..' in entry.parts:
                    continue
                source = Path(installed.locate_file(entry))
                if source.is_file() and source.suffix != '.pyc':
                    target = root.joinpath(*entry.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copy(source, target)
        if not (root / 'rsi' / 'roles.py').is_file():
            raise RuntimeError('RSI machinery inventory lacks roles worker')
        check()
        yield root
    check()


class IsolatedRoles:
    """Explicit roles Python executable; host SDK/environment stays untouched.

    The child runs an owned-package projection of this exact machinery via PYTHONPATH, using
    its own NOOA dependencies. Only the selected registered credential crosses
    in the controlled environment. JSON contains evidence and receipts, not keys.
    Observer role_finished receipts are authoritative actor elapsed time; worker
    loss emits accounting_incomplete and must suspend budgeted host dispatch.
    """

    def __init__(self, roles_python, profile, *, observer, environment,
                 setup_seconds=60, receipt_grace_seconds=2, clock=time.monotonic):
        from benchmarks.a5kernels.phase1_provider import direct_profile
        if profile != direct_profile(profile.identity):
            raise ValueError('unregistered role provider')
        executable = Path(roles_python)
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError('roles_python must be an explicit existing absolute executable')
        if type(setup_seconds) not in (int, float) or not math.isfinite(setup_seconds) or setup_seconds <= 0:
            raise ValueError('setup_seconds must be positive finite seconds')
        if (type(receipt_grace_seconds) not in (int, float) or not math.isfinite(receipt_grace_seconds)
                or receipt_grace_seconds < 0):
            raise ValueError('receipt_grace_seconds must be finite nonnegative seconds')
        self.receipt_grace_seconds = receipt_grace_seconds
        key = profile.identity.credential_env
        secret = environment.get(key)
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError('missing selected provider credential')
        self.environment = {k: environment[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT')
                            if k in environment}
        self.environment.update({key: secret, 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1'})
        self.roles_python, self.profile = str(executable), profile
        self.observer, self.setup_seconds, self.clock = observer, setup_seconds, clock

    def curriculum(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('curriculum', prompt, evidence, deadline, wall_deadline)

    def research_queries(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('research_queries', prompt, evidence, deadline, wall_deadline)

    def diagnostic(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('diagnostic', prompt, evidence, deadline, wall_deadline)

    def skillwriter(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('skillwriter', prompt, evidence, deadline, wall_deadline)

    def review(self, prompt, evidence, *, deadline, wall_deadline=None):
        return self._run('review', prompt, evidence, deadline, wall_deadline)

    def _run(self, role, prompt, evidence, deadline, wall_deadline=None):
        _wall_remaining(self.clock, wall_deadline)
        if type(deadline) not in (float, int) or not math.isfinite(deadline):
            raise ValueError('invalid actor deadline')
        allowance = deadline - self.clock()
        if allowance <= 0:
            raise TimeoutError('noncoding role actor time exhausted')
        evidence = list(evidence)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('role prompt must be nonempty')
        if any(not isinstance(e, AdmittedEvidence) or type(e.evidence_id) is not str or not e.evidence_id
               for e in evidence) or len({e.evidence_id for e in evidence}) != len(evidence):
            raise ValueError('admitted evidence requires unique nonempty IDs')
        request = {'role': role, 'prompt': prompt, 'evidence': [asdict(e) for e in evidence],
                   'backend_model': self.profile.identity.backend_model.value, 'seconds': allowance,
                   'wall_deadline': wall_deadline}
        wire = json.dumps(request, allow_nan=False)
        secret = self.environment[self.profile.identity.credential_env]
        if secret in wire:
            raise ValueError('credential cannot appear in role JSON')
        with _machinery_projection(lambda: _wall_remaining(self.clock, wall_deadline)) as root:
            result = self._dispatch(role, wire, allowance, evidence, secret, root, wall_deadline)
        _wall_remaining(self.clock, wall_deadline)
        return result

    def _dispatch(self, role, wire, allowance, evidence, secret, root, wall_deadline=None):
        _wall_remaining(self.clock, wall_deadline)
        started = self.clock()
        cutoff, active, finished, proposal = started + self.setup_seconds, False, False, None
        draining = False
        process = subprocess.Popen([self.roles_python, '-m', 'rsi.roles'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   env={**self.environment, 'PYTHONPATH': str(root)}, cwd=root)
        try:
            outgoing = wire.encode() + b'\n'
            os.set_blocking(process.stdin.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdin, selectors.EVENT_WRITE)
                selector.register(process.stdout, selectors.EVENT_READ)
                pending = b''
                while True:
                    effective = min(cutoff, wall_deadline) if wall_deadline is not None else cutoff
                    remaining = effective - self.clock()
                    ready = selector.select(max(0, remaining))
                    now = self.clock()
                    if now >= effective:
                        if (active and not finished and not draining and self.receipt_grace_seconds
                                and (wall_deadline is None or now < wall_deadline)):
                            # The worker's API deadline is unchanged. Collect only
                            # authoritative completion/accounting; never admit a late proposal.
                            draining = True
                            cutoff = now + self.receipt_grace_seconds
                        else:
                            raise TimeoutError('roles worker deadline exhausted')
                    if not ready:
                        continue
                    if any(key.fileobj is process.stdin for key, _ in ready):
                        sent = os.write(process.stdin.fileno(), outgoing[:65536])
                        outgoing = outgoing[sent:]
                        if not outgoing:
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    if not any(key.fileobj is process.stdout for key, _ in ready):
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    pending += chunk
                    if len(pending) > 4_000_000:
                        raise ValueError('oversized roles receipt')
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        if secret.encode() in line:
                            raise ValueError('credential in worker output')
                        message = json.loads(line)
                        if type(message) is not dict or set(message) != {'event', 'payload'}:
                            raise ValueError('invalid worker envelope')
                        event, payload = message['event'], message['payload']
                        if type(payload) is not dict:
                            raise ValueError('invalid worker payload')
                        if event in ('role_started', 'role_finished', 'role_response', 'role_proposal') and payload.get('role', role) != role:
                            raise ValueError('worker role mismatch')
                        if event == 'role_started':
                            if active or finished or payload['provider_identity'] != self.profile.identity.as_dict():
                                raise ValueError('invalid worker start receipt')
                            active = True
                            cutoff = self.clock() + allowance
                        elif event == 'role_finished':
                            elapsed = payload['elapsed_seconds']
                            if not active or finished or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or elapsed > self.clock() - started:
                                raise ValueError('invalid worker elapsed receipt')
                            if type(payload.get('success')) is not bool:
                                raise ValueError('invalid worker outcome')
                            finished = True
                            self.observer(event, payload)
                            if elapsed > allowance or draining:
                                raise TimeoutError('worker actor cutoff reached; actual elapsed receipt retained')
                            if not payload['success']:
                                error = payload.get('error')
                                if error is not None:
                                    if (type(error) is not dict or set(error) != {'type', 'message'}
                                            or any(type(v) is not str for v in error.values())
                                            or len(error['type'].encode()) > 128
                                            or len(error['message'].encode()) > 2048):
                                        raise ValueError('invalid bounded role failure detail')
                                    raise RoleFailure(role, error['type'], error['message'])
                                raise RuntimeError('role failed; elapsed receipt retained')
                            cutoff = self.clock() + self.setup_seconds
                        elif event == 'result':
                            if not finished or proposal is not None:
                                raise ValueError('unordered worker result')
                            proposal = _proposal(role, payload, {e.evidence_id for e in evidence})
                        elif event == 'worker_error':
                            raise RuntimeError('roles worker failed; inspect host receipts')
                        elif event not in ('request', 'rejection', 'role_response', 'role_proposal') or not active or finished:
                            raise ValueError('unordered or unknown worker receipt')
                        if event != 'role_finished':
                            self.observer(event, payload)
            _wall_remaining(self.clock, wall_deadline)
            effective = min(cutoff, wall_deadline) if wall_deadline is not None else cutoff
            process.wait(timeout=max(0.001, effective - self.clock()))
            if process.returncode or not finished or proposal is None or pending:
                raise RuntimeError('roles worker missing completion receipts')
            return proposal
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
            if not finished:
                self.observer('accounting_incomplete', {'role': role, 'start_receipt_received': active})


def _worker():
    """Private JSON-lines worker, reusing the registered provider and role path."""
    import contextlib
    protocol = sys.stdout
    secret = None

    def emit(event, payload):
        line = json.dumps({'event': event, 'payload': payload}, allow_nan=False)
        if secret and secret in line:
            raise ValueError('credential in role output')
        protocol.write(line + '\n')
        protocol.flush()

    try:
        request = json.loads(sys.stdin.readline())
        from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
        from benchmarks.a5kernels.phase1_provider import DirectCompletionProvider, direct_profile
        identity = next(x for x in MODEL_IDENTITIES if x.backend_model.value == request['backend_model'])
        secret = os.environ[identity.credential_env]
        with contextlib.redirect_stdout(sys.stderr):
            provider = DirectCompletionProvider({identity.credential_env: secret},
                                                models=[identity.backend_model], invalid_prompt_retries=0)
            roles = NoncodingRoles(provider, direct_profile(identity), observer=emit)
            evidence = [AdmittedEvidence(**e) for e in request['evidence']]
            if request['role'] not in _PROPOSALS:
                raise ValueError('unknown role')
            result = getattr(roles, request['role'])(request['prompt'], evidence,
                                                    deadline=time.monotonic() + request['seconds'],
                                                    wall_deadline=request.get('wall_deadline'))
        emit('result', asdict(result))
        return 0
    except Exception:
        emit('worker_error', {'reason': 'role_failed'})
        return 1


if __name__ == '__main__':
    raise SystemExit(_worker())
