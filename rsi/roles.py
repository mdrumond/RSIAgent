"""Typed, proposal-only NOOA roles over the host's registered direct provider.

Importing this module needs only the standard library. Install ``rsi[roles]``
with Python 3.12/3.13 to invoke a role. The host supplies admitted evidence,
absolute monotonic deadlines, and an observer for artifacts/accounting. Calls
are synchronous, including the bridge's async entry, so cancellation cannot
leave an unobserved provider thread running. Provider transport enforces timeout.
No tools, execution, memory storage, or proposal application is delegated here.
"""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import sys
import time
from typing import Any, Literal, Mapping


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
    operation: Literal['replace', 'remove']
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

    class Edit(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        skill: str = Field(min_length=1)
        operation: Literal['replace', 'remove']
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

    def curriculum(self, role_prompt, evidence, *, deadline):
        return self._run('curriculum', role_prompt, evidence, deadline)

    def skillwriter(self, role_prompt, evidence, *, deadline):
        return self._run('skillwriter', role_prompt, evidence, deadline)

    def review(self, role_prompt, evidence, *, deadline):
        return self._run('review', role_prompt, evidence, deadline)

    def _run(self, role, prompt, evidence, deadline):
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
        deadline = client.deadline = started + allowance
        success = False
        self.observer('role_started', {'role': role, 'provider_identity': self.profile.identity.as_dict(),
                                       'prompt': prompt, 'evidence': rows})
        try:
            result = asyncio.run(getattr(agent, role)(role_prompt=prompt, admitted_evidence=encoded))
            self._remaining(deadline)
            if set(result.evidence_ids) - ids:
                raise ValueError('proposal cites evidence not admitted by the host')
            common = (result.rationale, tuple(result.evidence_ids))
            if role == 'curriculum':
                proposal = CurriculumProposal(*common, tuple(ProjectIdea(**x.model_dump()) for x in result.projects))
            elif role == 'skillwriter':
                proposal = SkillProposal(*common, tuple(SkillEdit(**x.model_dump()) for x in result.edits))
            else:
                proposal = ReviewProposal(*common, tuple(ReviewFinding(**x.model_dump()) for x in result.findings))
            self.observer('role_proposal', {'role': role, 'proposal': result.model_dump(mode='json')})
            success = True
            return proposal
        finally:
            self.observer('role_finished', {'role': role, 'success': success,
                                           'elapsed_seconds': self.clock() - started})
