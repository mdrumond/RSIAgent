"""Independent evidence-reading project review and versioned memory proposals.

Callbacks receive ``remaining_seconds`` and must bound their own execution.
The verifier never writes active memory; publication belongs to its caller.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
import math
import re
import time


_SHA = re.compile(r'[0-9a-f]{64}\Z')


class EvidenceReferenceError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _digest(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError('artifact identity must be SHA-256')
    return value


def artifact_preview(registry, limit=16):
    """Prioritize core review evidence without hiding the paged registry."""
    def priority(role):
        parts = role.lower().split(':')
        core = ('source', 'result', 'execution', 'log', 'evidence', 'contract')
        if role == 'source':
            return (-1, role)
        return (next((i for i, word in enumerate(core)
                      if any(word in part for part in parts)), len(core)), role)
    return dict(sorted(registry.items(), key=lambda item: priority(item[0]))[:limit])


def artifact_page(registry, payload):
    """List stable role names; newly observed tools may extend the registry."""
    if not isinstance(payload, dict) or set(payload) - {'offset', 'limit', 'prefix'}:
        raise ValueError('invalid artifact listing')
    offset, limit, prefix = payload.get('offset', 0), payload.get('limit', 20), payload.get('prefix', '')
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50 or not isinstance(prefix, str):
        raise ValueError('artifact listing needs offset >= 0, limit 1..50 and text prefix')
    rows = sorted((role, digest) for role, digest in registry.items() if role.startswith(prefix))
    end = offset + limit
    return dict(artifacts=dict(rows[offset:end]), total=len(rows),
                next_offset=end if end < len(rows) else None)


def knowledge_preview(observation, registry):
    """Retain exact version/citation identities while paging rich documents."""
    memories = []
    for memory in observation.get('memories', []):
        prefix = 'memory:' + memory['memory_id'] + ':' + memory['version'] + ':'
        registry.update({prefix + role: digest for role, digest in memory.get('artifacts', {}).items()})
        proposal = memory.get('proposal', memory)
        memories.append(dict(memory_id=memory['memory_id'], version=memory['version'],
            **{key: str(proposal.get(key, ''))[:512] for key in ('summary', 'preconditions', 'expectations')},
            artifacts=artifact_preview(memory.get('artifacts', {}), 4),
            artifact_count=len(memory.get('artifacts', {})), artifact_prefix=prefix))
    hits = [dict(citation=hit.get('citation'), score=hit.get('score'),
                 text=str(hit.get('text', ''))[:1024]) for hit in observation.get('hits', [])[:5]]
    return dict(memories=memories, citations=observation.get('citations', []), hits=hits)


@dataclass(frozen=True)
class ProjectReview:
    project_id: str
    source_sha256: str | None
    host_verified: bool
    artifacts: Mapping[str, str]
    host_facts: tuple = ()
    objective: str = ''
    memory_exposures: tuple = ()
    memory_repairs: tuple = ()
    retired_memories: tuple = ()

    def __post_init__(self):
        _text(self.project_id, 'project_id')
        if type(self.host_verified) is not bool:
            raise ValueError('host_verified must be a host-owned boolean')
        if self.source_sha256 is None:
            if self.host_verified:
                raise ValueError('host-verified projects require retained source')
        else:
            _digest(self.source_sha256)
        for role, digest in self.artifacts.items():
            _text(role, 'artifact role')
            _digest(digest)
        # Keep caller-owned mutable dictionaries out of a frozen request.
        object.__setattr__(self, 'artifacts', dict(self.artifacts))


SYSTEM = '''You are an independent project verifier, separate from the Actor.
Inspect code, retained results and logs, and host facts. A passing goal can be
accidental or insufficiently thorough: request controls/variants or send the
project to curriculum repair with precise instructions and expected evidence.
Never override a failed host check. Evaluate each project's own frozen contract,
including its stated control/probe role, within the broader subject objective.
A working baseline control need not itself implement the subject's final operation.
Query related KDB memories when available;
read relevant prior artifacts before proposing an append or merge. Keep facts
separate from interpretation and state applicability, preconditions, expected
behavior and limitations. Explain why evidence supports the lesson, not just
that the original project passed. A rewrite feedback request concerns memory:
rewrite using the retained project evidence; request project repair if the
underlying evidence is insufficient.
'''

ACTION_FORMAT = '''Return only one action JSON object per reply. Never add API configuration keys such as type or response_format.
Return one JSON action per reply:
{"list_artifacts":{"offset":0,"limit":20,"prefix":"optional role prefix"}}
{"read_artifact":{"sha256":"...","offset":0,"limit":8192}}
{"knowledge_query":{"query":"..."}}
{"request_check":{"hypothesis":"...","check":{...}}}
or terminal {"decision":"accept"|"repair","reason":"...","evidence":["sha"],
"memory_judgments":[{"exposure_id":"...","usefulness":"useful|not-useful|unknown",
"correctness":"supported|contradicted|unknown","investigate":false,"reason":"...","evidence":["sha"]}],
"memory_resolutions":[{"repair_id":"...","action":"replace|restore|retire","reason":"...","evidence":["sha"]}],
"retirement_checks":[{"tombstone_id":"...","relation":"unrelated|duplicate|reconciles","reason":"...","evidence":["sha"]}],
"repair":null|{"instructions":"...","expected_evidence":"..."},"memory":null|{
"operation":"create"|"append"|"merge","base_versions":[{"memory_id":"...","version":"..."}],
"summary":"...","applicability":"...","preconditions":"...","expectations":"...",
"limitations":"...","before":null|{"sha256":"...","start_line":1,"end_line":2,"text":"exact lines"},
"before_unavailable_reason":null|"...",
"after":{"sha256":"...","start_line":1,"end_line":2,"text":"exact lines"},
"evidence":["sha"],"citations":[exact KDB citation objects]}}.
'''

EVIDENCE_INSTRUCTIONS = '''Read the current source and supplied result/log artifacts before acceptance.
All snippets must be exact lines from artifacts actually read, including line
endings. Prefer omitting the optional snippet text field: select sha256/start_line/
end_line from source you have read, and the host will extract those exact lines.
Do not double-escape code or guess line endings. If text is supplied it must
match exactly. A snippet formatting failure is not a reason to discard an
otherwise useful lesson; correct its range or omit text. Use a null before plus
a reason if no distinct retained before-code exists. When original-source differs
from current source, read its complete contents using missing_required_artifacts
and returned next_offset, then preserve an exact snippet from that original as
before. Never substitute current source or an unrelated KDB example for it.
An original-source artifact is historical code, not proof of failure. Preserve
known passing outcomes and distinguish untested code from a reproduced failure;
do not invent a failed-before or causal repair claim. Readable artifacts are
available to read; allowed_evidence contains only artifacts actually observed,
while fully_read_artifacts distinguishes complete reads from partial pages.
Append/merge bases must be exact versions returned by KDB. An append has one
base; a merge has one or more. A create has none. For an accepted project, normally compose a reusable memory with the verified
example, its expected behavior, applicability and preconditions. New means not
already represented in the accepted memory catalog; standard APIs and techniques
can still warrant a new host-validated example. An empty KDB result is a reason
to CREATE relevant memory, never a prerequisite failure for writing memory.
Acceptance may have null memory only when existing accepted memory already covers
the learning or the project yielded no reusable learning; explain that specific
reason. Usefulness is assessed from ordinary project evidence, so do not suppress useful
verified examples merely because the API is familiar. Repair must have null memory.
Evidence must reference actually read artifacts; no unsupported causal claims.
Judge EVERY supplied memory_exposure independently for usefulness and correctness.
Use unknown when traces/results cannot establish either; unknown is not negative evidence.
Set investigate=true only for an explicit cited correctness concern; ordinary uncertainty
from missing evidence is not an investigation request.
Read its exact document before deciding correctness; contradiction must cite separate
project evidence. A correct memory can be unhelpful, and an incorrect memory can appear
helpful. Resolve assigned memory_repairs explicitly: replace with a corrected/merged
memory, restore an unjustified quarantine, or retire with evidence. A passing kernel
alone never resolves a memory repair. Compare proposed memories against all supplied
retired_memories; duplicate retired claims cannot be republished. Reconciliation needs
fresh project evidence explicitly resolving the recorded retirement evidence.
All three memory arrays are required; use [] when no entries apply.'''

REVIEW_INSTRUCTIONS = SYSTEM
SYSTEM = REVIEW_INSTRUCTIONS + ACTION_FORMAT + EVIDENCE_INSTRUCTIONS


def bounded_review_messages(messages, state, max_bytes=40000):
    """Keep instructions and exact review state while paging older raw outputs."""
    def size(rows):
        return len(json.dumps(rows, ensure_ascii=False).encode('utf-8'))
    if size(messages) <= max_bytes:
        return messages
    pinned = messages[:2] + [{'role': 'user', 'content': json.dumps(dict(
        retained_review_state=state,
        note='Older raw pages were folded to fit context. Read artifacts again when exact contents are needed.'), sort_keys=True)}]
    tail = []
    for end in range(len(messages), 2, -2):
        pair = messages[max(2,end-2):end]
        if size(pinned + pair + tail) > max_bytes:
            break
        tail = pair + tail
    return pinned + tail


class ProjectVerifierAgent:
    def __init__(self, complete, read_artifact, *, query_knowledge=None,
                 request_check=None, clock=time.monotonic, trace=None, session_factory=None):
        self.complete, self.read_artifact = complete, read_artifact
        self.query_knowledge, self.request_check = query_knowledge, request_check
        self.clock, self.trace = clock, trace
        self.session_factory = session_factory
        self.last_trace = []

    def _record(self, kind, payload):
        event = {'kind': kind, 'payload': payload}
        self.last_trace.append(event)
        if self.trace is not None:
            self.trace(event)

    def review(self, request: ProjectReview, *, max_seconds=600, feedback=None):
        if not isinstance(request, ProjectReview):
            raise TypeError('review requires ProjectReview')
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('max_seconds must be finite and positive')
        self.last_trace = []
        deadline = self.clock() + max_seconds

        def remaining():
            seconds = deadline - self.clock()
            if seconds <= 0:
                raise TimeoutError('project verifier elapsed allocation exhausted')
            return seconds

        with ExitStack() as stack:
            session = (stack.enter_context(self.session_factory(max_seconds=remaining()))
                       if self.session_factory is not None else None)
            return self._review(request, remaining, feedback, session)

    def _review(self, request, remaining, feedback, session):
        state = ReviewState(self, request, remaining, feedback)
        if session is not None and hasattr(session, 'review'):
            return session.review(state)
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': json.dumps(state.context, sort_keys=True)}]
        while True:
            if session is None:
                response = self.complete(tuple(bounded_review_messages(messages, state.summary())),
                                         remaining_seconds=remaining())
            else:
                from coding.pi.session import ActionError
                try:
                    name, args = session.next_action(dict(project=state.context, review_state=state.summary(),
                        artifact_pages=list(state.pages.values()), tool_observations=state.tools,
                        validation_feedback=state.feedback), state.allowed(), record=self._record)
                    response = json.dumps(args if name == 'submit_review' else {name: args})
                except ActionError as error:
                    response = json.dumps({'invalid_typed_action': str(error)})
            remaining()
            messages.append({'role': 'assistant', 'content': response})
            try:
                value = json.loads(response)
            except (ValueError, TypeError):
                value = None
            observation = state.execute(value, raw_response=response)
            if state.decision is not None:
                return state.decision
            messages.append({'role': 'user', 'content': json.dumps(observation, sort_keys=True)})

    def _validate_decision(self, value, request, read, required, host_ok, queried,
                           versions, citations, remaining, observed, fully_read):
        if set(value) != {'decision', 'reason', 'evidence', 'repair', 'memory', 'memory_judgments', 'memory_resolutions', 'retirement_checks'} or value['decision'] not in {'accept', 'repair'}:
            raise ValueError('invalid terminal decision')
        _text(value['reason'], 'reason')
        self._evidence(value['evidence'], read)
        from .memory_review import validate_memory_review
        validate_memory_review(value, request, read, self._evidence, fully_read)
        if value['decision'] == 'repair':
            if request.source_sha256 is None and required - read:
                raise ValueError('read execution result/log evidence for a project without source')
            if value['memory'] is not None or not isinstance(value['repair'], dict) or set(value['repair']) != {'instructions', 'expected_evidence'}:
                raise ValueError('repair needs instructions/expected_evidence and no memory')
            for key, text in value['repair'].items():
                _text(text, key)
            return
        if not host_ok:
            raise ValueError('failed host checks require repair')
        if required - read:
            raise ValueError('read current source and result/log evidence before acceptance')
        if not queried:
            raise ValueError('query related KDB memories before acceptance')
        if value['repair'] is not None:
            raise ValueError('acceptance cannot carry repair instructions')
        memory = value['memory']
        if memory is None:
            return
        fields = {'operation', 'base_versions', 'summary', 'applicability', 'preconditions',
                  'expectations', 'limitations', 'before', 'before_unavailable_reason',
                  'after', 'evidence', 'citations'}
        if not isinstance(memory, dict) or set(memory) != fields or memory['operation'] not in {'create', 'append', 'merge'}:
            raise ValueError('invalid memory proposal')
        for field in ('summary', 'applicability', 'preconditions', 'expectations', 'limitations'):
            _text(memory[field], field)
        bases = memory['base_versions']
        if not isinstance(bases, list) or any(not isinstance(base, dict) or set(base) != {'memory_id', 'version'} for base in bases):
            raise ValueError('memory bases require memory_id/version')
        if len({base['memory_id'] for base in bases}) != len(bases):
            raise ValueError('memory bases must identify distinct memories')
        if any((base['memory_id'], base['version']) not in versions for base in bases):
            raise ValueError('memory base was not returned by KDB')
        if (memory['operation'] == 'create' and bases or memory['operation'] == 'append' and len(bases) != 1
                or memory['operation'] == 'merge' and not bases):
            raise ValueError('memory base count disagrees with operation')
        self._evidence(memory['evidence'], read)
        if not isinstance(memory['citations'], list) or any(item not in citations for item in memory['citations']):
            raise ValueError('memory citation was not returned by KDB')
        original = request.artifacts.get('original-source')
        distinct_original = original is not None and original != request.source_sha256
        if memory['before'] is None:
            if distinct_original:
                raise EvidenceReferenceError('retained-before-required',
                    f'retained original-source {original} exists; preserve its exact before snippet '
                    'and set before_unavailable_reason to null; do not claim it is absent')
            _text(memory['before_unavailable_reason'], 'before_unavailable_reason')
        elif memory['before_unavailable_reason'] is not None:
            raise ValueError('before snippet cannot have an unavailable reason')
        else:
            self._snippet(memory['before'], read, remaining, observed)
        self._snippet(memory['after'], read, remaining, observed)
        if memory['after']['sha256'] != request.source_sha256:
            raise ValueError('after snippet must reference reviewed current source')
        if memory['before'] is not None:
            if memory['before']['sha256'] == memory['after']['sha256']:
                raise EvidenceReferenceError('identical-before-after',
                    'before and after reference identical source; use the distinct retained '
                    'original-source, or null before with an honest reason when none exists')
            if distinct_original and memory['before']['sha256'] != original:
                raise EvidenceReferenceError('wrong-before-source',
                    f"before must reference this project's retained original-source {original}, "
                    'not a different example returned by KDB')

    @staticmethod
    def _evidence(items, read):
        if not isinstance(items, list) or not items:
            raise EvidenceReferenceError('malformed-evidence',
                'evidence must be a nonempty list of bare SHA256 artifact identities')
        if any(not isinstance(item, str) or not _SHA.fullmatch(item) for item in items):
            raise EvidenceReferenceError('malformed-evidence',
                'malformed evidence reference: use bare SHA256 artifact identities, not prose')
        if any(item not in read for item in items):
            raise EvidenceReferenceError('unread-evidence', 'read cited artifacts before acceptance; '
                             f'unread references: {[item for item in items if item not in read]}')

    def _snippet(self, snippet, read, remaining, observed):
        if not isinstance(snippet, dict) or set(snippet) not in ({'sha256', 'start_line', 'end_line'},
                                                               {'sha256', 'start_line', 'end_line', 'text'}):
            raise ValueError('invalid source snippet')
        if snippet['sha256'] not in read:
            raise ValueError('read source artifact before quoting a snippet')
        data = self.read_artifact(snippet['sha256'], remaining_seconds=remaining())
        if hashlib.sha256(data).hexdigest() != snippet['sha256']:
            raise ValueError('source artifact bytes changed')
        lines = data.decode('utf-8').splitlines(keepends=True)
        start, end = snippet['start_line'], snippet['end_line']
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError('snippet lines are outside source artifact')
        exact = ''.join(lines[start - 1:end])
        if 'text' in snippet and snippet['text'] != exact:
            raise ValueError('snippet must match exact retained source lines')
        # A private full-artifact validation read is not an observation by the
        # agent. Quoted bytes must have appeared in one or more returned pages.
        quote_start = len(''.join(lines[:start - 1]).encode('utf-8'))
        quote_end = quote_start + len(exact.encode('utf-8'))
        covered = quote_start
        for page_start, page_end in sorted(observed.get(snippet['sha256'], [])):
            if page_start > covered:
                break
            covered = max(covered, page_end)
            if covered >= quote_end:
                snippet['text'] = exact
                return
        raise ValueError('read all quoted source bytes before quoting a snippet')


class ReviewState:
    """One host-owned decision/read contract shared by completion and native Pi."""
    def __init__(self, agent, request, remaining, feedback):
        self.agent, self.request, self.remaining = agent, request, remaining
        self.registry = dict(request.artifacts)
        if request.source_sha256 is not None:
            self.registry['source'] = request.source_sha256
        for group in (request.memory_exposures, request.memory_repairs, request.retired_memories):
            for row in group:
                self.registry['memory-document:' + row['version']] = row['document_sha256']
                for index, sha in enumerate({*row.get('evidence', []), *row.get('artifact_sha256', [])}):
                    self.registry['memory-evidence:' + row['version'] + ':' + str(index)] = sha
        self.known = set(self.registry.values())
        self.read, self.observed, self.versions, self.citations = set(), {}, set(), []
        for row in (*request.memory_repairs, *request.retired_memories):
            self.versions.add((row['memory_id'], row['version']))
        self.sizes = {}
        original = request.artifacts.get('original-source')
        self.complete_required = {original} if original is not None and original != request.source_sha256 else set()
        self.queried, self.host_ok = agent.query_knowledge is None, request.host_verified
        self.execution_required = {digest for role, digest in request.artifacts.items()
            if any(part in role.lower() for part in ('result', 'log', 'evidence', 'execution'))}
        self.required = set(self.execution_required)
        if request.source_sha256 is not None:
            self.required.add(request.source_sha256)
        self.required.update(self.complete_required)
        project = asdict(request)
        project.update(artifacts=artifact_preview(self.registry), artifact_count=len(self.registry))
        self.context = dict(project=project, rewrite_feedback=feedback,
            tools=dict(knowledge_query=agent.query_knowledge is not None,
                       request_check=agent.request_check is not None))
        agent._record('request', dict(self.context, project=asdict(request)))
        self.pages, self.tools, self.cache, self.read_requests = {}, {}, {}, {}
        self.native, self.feedback, self.decision, self.raw_observation = False, None, None, None

    def roles(self, digest):
        return sorted(role for role, value in self.registry.items() if value == digest)

    def allowed(self):
        return ['list_artifacts', 'read_artifact', 'submit_review'] + (
            ['knowledge_query'] if self.agent.query_knowledge is not None else []) + (
            ['request_check'] if self.agent.request_check is not None else [])

    def missing_ranges(self, digest):
        size = self.sizes.get(digest)
        if size is None:
            return [dict(offset=0, end=None)]
        position, gaps = 0, []
        for start, end in sorted(self.observed.get(digest, [])):
            if start > position:
                gaps.append(dict(offset=position, end=start))
            position = max(position, end)
        if position < size:
            gaps.append(dict(offset=position, end=size))
        return gaps

    def missing_required(self):
        missing = self.required - self.read
        missing |= {digest for digest in self.complete_required if self.missing_ranges(digest)}
        rows = []
        for digest in sorted(missing):
            row = dict(sha256=digest, roles=self.roles(digest))
            if digest in self.complete_required:
                gaps = self.missing_ranges(digest)
                row.update(complete_read_required=True, missing_ranges=gaps,
                    next_read=dict(sha256=digest, offset=gaps[0]['offset'],
                                   limit=2048 if self.native else 8192))
            rows.append(row)
        return rows

    def readable_preview(self):
        return sorted(set(artifact_preview(self.registry).values()))

    def summary(self):
        missing = self.missing_required()
        ready = not missing and self.queried
        return dict(read_artifacts=sorted(self.read), allowed_evidence=sorted(self.read),
            readable_artifacts=self.readable_preview(), readable_artifact_count=len(self.known),
            fully_read_artifacts=sorted(digest for digest in self.read if not self.missing_ranges(digest)),
            missing_required_artifacts=missing,
            observed_versions=sorted(self.versions), citations=self.citations,
            host_checks_passed=self.host_ok, knowledge_queried=self.queried,
            ready_for_decision=ready,
            read_progress=[dict(sha256=h, roles=self.roles(h), observed_ranges=self.observed[h],
                requests=sum(n for (digest, _, _), n in self.read_requests.items() if digest == h))
                for h in sorted(self.read)] if self.native else [],
            guidance=('Evidence prerequisites are complete; submit a justified accept or repair when ready. '
                      'Further reads remain available; list_artifacts pages the complete readable registry.' if ready else
                      'Read the missing evidence and query related memory; list_artifacts pages the complete readable registry.'))

    def execute(self, value, *, raw_response=None):
        self.remaining()
        self.raw_observation = None
        self.agent._record('response', json.dumps(value) if raw_response is None else raw_response)
        registry, known, read, observed = self.registry, self.known, self.read, self.observed
        versions, citations, required = self.versions, self.citations, self.required
        queried, host_ok, request, remaining = self.queried, self.host_ok, self.request, self.remaining
        try:
            if not isinstance(value, dict):
                raise ValueError('reply must be a JSON object')
            if set(value) == {'list_artifacts'}:
                observation = artifact_page(registry, value['list_artifacts'])
            elif set(value) == {'read_artifact'}:
                payload = value['read_artifact']
                if not isinstance(payload, dict) or set(payload) - {'sha256', 'offset', 'limit'}:
                    raise ValueError('invalid artifact read')
                digest = _digest(payload['sha256'])
                if digest not in known:
                    raise ValueError('artifact is not admitted by the project or observed tools')
                cached = self.native and digest in self.cache
                try:
                    data = self.cache[digest] if cached else self.agent.read_artifact(digest, remaining_seconds=remaining())
                except OSError as error:
                    if isinstance(error, TimeoutError):
                        raise
                    raise EvidenceReferenceError('artifact-unavailable',
                        f'admitted artifact {digest} is unreadable ({type(error).__name__}); '
                        'restore its retained bytes or request repair; this is not evidence of absence') from error
                if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('artifact bytes do not match their identity')
                if self.native:
                    self.cache[digest] = data
                offset, limit = payload.get('offset', 0), payload.get('limit', 2048 if self.native else 8192)
                if type(offset) is not int or not 0 <= offset <= len(data) or type(limit) is not int or not 1 <= limit <= 65536:
                    raise ValueError('invalid artifact page')
                end = min(len(data), offset + min(limit, 2048 if self.native else 8192))
                if self.native:
                    if offset < len(data) and offset and data[offset] & 0xC0 == 0x80:
                        raise ValueError('offset must be a returned UTF-8 page boundary')
                    while end < len(data) and end > offset and data[end] & 0xC0 == 0x80:
                        end -= 1
                    if end == offset and offset < len(data):
                        raise ValueError('page limit is too small for the next UTF-8 character')
                page = data[offset:end]
                self.sizes[digest] = len(data)
                if self.native:
                    # Keep escaped JSON page envelopes bounded as well as bytes.
                    while len(json.dumps(dict(text=page.decode('utf-8', errors='replace')),
                                         ensure_ascii=False).encode()) > 3000:
                        end = offset + len(page)//2
                        while end > offset and data[end] & 0xC0 == 0x80:
                            end -= 1
                        page = data[offset:end]
                if page or not data:
                    read.add(digest)
                    observed.setdefault(digest, []).append((offset, offset + len(page)))
                    if self.native:
                        merged = []
                        for first, last in sorted(observed[digest]):
                            if merged and first <= merged[-1][1]:
                                merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
                            else:
                                merged.append((first, last))
                        observed[digest] = merged
                observation = {'sha256': digest, 'offset': offset, 'size': len(data),
                               'text': page.decode('utf-8', errors='replace'),
                               'next_offset': offset + len(page) if offset + len(page) < len(data) else None}
                if self.native:
                    key = (digest, offset, len(page))
                    repeat = key in self.read_requests
                    self.read_requests[key] = self.read_requests.get(key, 0) + 1
                    observation.update(cache_hit=cached, repeat=repeat, page_requests=self.read_requests[key])
                    self.agent._record('review-read', dict(sha256=digest, offset=offset, bytes=len(page),
                        cache_hit=cached, repeat=repeat, roles=self.roles(digest)))
            elif set(value) == {'knowledge_query'}:
                payload = value['knowledge_query']
                if self.agent.query_knowledge is None:
                    raise ValueError('knowledge query is unavailable')
                if not isinstance(payload, dict) or set(payload) != {'query'}:
                    raise ValueError('knowledge query requires query')
                _text(payload['query'], 'query')
                observation = self.agent.query_knowledge(payload, remaining_seconds=remaining())
                if not isinstance(observation, Mapping):
                    raise ValueError('knowledge result must be a mapping')
                for memory in observation.get('memories', []):
                    versions.add((_text(memory['memory_id'], 'memory_id'),
                                  _text(memory['version'], 'version')))
                    for digest in memory.get('artifacts', {}).values():
                        known.add(_digest(digest))
                citations.extend(observation.get('citations', []))
                self.raw_observation = observation
                observation = knowledge_preview(observation, registry)
                queried = True
            elif set(value) == {'request_check'}:
                if self.agent.request_check is None:
                    raise ValueError('additional checks are unavailable')
                payload = value['request_check']
                if not isinstance(payload, dict) or set(payload) != {'hypothesis', 'check'} or not isinstance(payload['check'], dict):
                    raise ValueError('check requires hypothesis and check object')
                _text(payload['hypothesis'], 'hypothesis')
                observation = self.agent.request_check(payload, remaining_seconds=remaining())
                if not isinstance(observation, Mapping) or type(observation.get('host_verified')) is not bool:
                    raise ValueError('check must return host_verified and artifact references')
                host_ok = host_ok and observation['host_verified']
                registry.update({'check:' + str(len(registry)) + ':' + role: digest
                                 for role, digest in observation.get('artifacts', {}).items()})
                for digest in observation.get('artifacts', {}).values():
                    required.add(_digest(digest))
                    self.execution_required.add(digest)
                    known.add(digest)
                observation = dict(host_verified=observation['host_verified'],
                                   artifacts=artifact_preview(registry), artifact_count=len(registry))
            else:
                if value.get('decision') == 'accept' and any(self.missing_ranges(h) for h in self.complete_required):
                    raise EvidenceReferenceError('incomplete-original-source',
                        'read the complete distinct original-source before acceptance; '
                        'follow missing_required_artifacts.next_read to read every missing page')
                decision_required = self.execution_required if value.get('decision') == 'repair' else required
                self.agent._validate_decision(value, request, read, decision_required, host_ok,
                                        queried, versions, citations, remaining, observed,
                                        {sha for sha in read if not self.missing_ranges(sha)})
                remaining()
                self.agent._record('decision', value)
                self.decision = value
                return value
            remaining()
            self.agent._record('tool', {'action': value, 'observation': observation})
            if 'read_artifact' in value:
                key = (observation['sha256'], observation['offset'])
                self.pages.pop(key, None)
                self.pages[key] = observation
                while len(self.pages) > 8 or (len(self.pages) > 1 and
                        len(json.dumps(list(self.pages.values())).encode()) > 12000):
                    self.pages.pop(next(iter(self.pages)))
            else:
                self.tools[next(iter(value))] = observation
            return observation
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
            observation = dict(error=str(error), error_code=getattr(error, 'code', 'invalid-review-action'),
                missing_required_artifacts=self.missing_required(), readable_artifacts=self.readable_preview(),
                readable_artifact_count=len(known),
                knowledge_query_required=not queried, allowed_evidence=sorted(read), allowed_citations=citations,
                correction='Use exact allowed evidence hashes/citations, 1-based source lines, and read missing evidence.')
            self.feedback = observation
            self.agent._record('feedback', observation)
            return observation
        finally:
            self.queried, self.host_ok = queried, host_ok
