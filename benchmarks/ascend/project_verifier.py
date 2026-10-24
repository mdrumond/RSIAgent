"""Independent evidence-reading project review and versioned memory proposals.

Callbacks receive ``remaining_seconds`` and must bound their own execution.
The verifier never writes active memory; publication belongs to its caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import hashlib
import json
import math
import re
import time


_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _digest(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError('artifact identity must be SHA-256')
    return value


@dataclass(frozen=True)
class ProjectReview:
    project_id: str
    source_sha256: str | None
    host_verified: bool
    artifacts: Mapping[str, str]
    host_facts: tuple = ()
    objective: str = ''

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
Return only one action JSON object per reply. Never add API configuration keys such as type or response_format.
Return one JSON action per reply:
{"read_artifact":{"sha256":"...","offset":0,"limit":8192}}
{"knowledge_query":{"query":"..."}}
{"request_check":{"hypothesis":"...","check":{...}}}
or terminal {"decision":"accept"|"repair","reason":"...","evidence":["sha"],
"repair":null|{"instructions":"...","expected_evidence":"..."},"memory":null|{
"operation":"create"|"append"|"merge","base_versions":[{"memory_id":"...","version":"..."}],
"summary":"...","applicability":"...","preconditions":"...","expectations":"...",
"limitations":"...","before":null|{"sha256":"...","start_line":1,"end_line":2,"text":"exact lines"},
"before_unavailable_reason":null|"...",
"after":{"sha256":"...","start_line":1,"end_line":2,"text":"exact lines"},
"evidence":["sha"],"citations":[exact KDB citation objects]}}.
Read the current source and supplied result/log artifacts before acceptance.
All snippets must be exact lines from artifacts actually read, including line
endings. Use a null before plus a reason if no retained before-code exists.
Append/merge bases must be exact versions returned by KDB. An append has one
base; a merge has one or more. A create has none. For an accepted project, normally compose a reusable memory with the verified
example, its expected behavior, applicability and preconditions. New means not
already represented in the accepted memory catalog; standard APIs and techniques
can still warrant a new host-validated example. An empty KDB result is a reason
to CREATE relevant memory, never a prerequisite failure for writing memory.
Acceptance may have null memory only when existing accepted memory already covers
the learning or the project yielded no reusable learning; explain that specific
reason. Usefulness will be tested independently, so do not suppress useful
verified examples merely because the API is familiar. Repair must have null memory.
Evidence must reference actually read artifacts; no unsupported causal claims.'''


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
                 request_check=None, clock=time.monotonic, trace=None):
        self.complete, self.read_artifact = complete, read_artifact
        self.query_knowledge, self.request_check = query_knowledge, request_check
        self.clock, self.trace = clock, trace
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

        known = set(request.artifacts.values())
        if request.source_sha256 is not None:
            known.add(request.source_sha256)
        read = set()
        observed = {}
        versions, citations = set(), []
        queried = self.query_knowledge is None
        host_ok = request.host_verified
        required = {digest for role, digest in request.artifacts.items()
                    if any(part in role.lower() for part in ('result', 'log', 'evidence', 'execution'))}
        if request.source_sha256 is not None:
            required.add(request.source_sha256)
        context = {'project': asdict(request), 'rewrite_feedback': feedback,
                   'tools': {'knowledge_query': self.query_knowledge is not None,
                             'request_check': self.request_check is not None}}
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': json.dumps(context, sort_keys=True)}]
        self._record('request', context)
        while True:
            working = bounded_review_messages(messages, dict(read_artifacts=sorted(read),
                observed_versions=sorted(versions), citations=citations, host_checks_passed=host_ok,
                knowledge_queried=queried))
            response = self.complete(tuple(working), remaining_seconds=remaining())
            remaining()
            self._record('response', response)
            messages.append({'role': 'assistant', 'content': response})
            try:
                value = json.loads(response)
                if not isinstance(value, dict):
                    raise ValueError('reply must be a JSON object')
                if set(value) == {'read_artifact'}:
                    payload = value['read_artifact']
                    if not isinstance(payload, dict) or set(payload) - {'sha256', 'offset', 'limit'}:
                        raise ValueError('invalid artifact read')
                    digest = _digest(payload['sha256'])
                    if digest not in known:
                        raise ValueError('artifact is not admitted by the project or observed tools')
                    data = self.read_artifact(digest, remaining_seconds=remaining())
                    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError('artifact bytes do not match their identity')
                    offset, limit = payload.get('offset', 0), payload.get('limit', 8192)
                    if type(offset) is not int or not 0 <= offset <= len(data) or type(limit) is not int or not 1 <= limit <= 65536:
                        raise ValueError('invalid artifact page')
                    page = data[offset:offset + min(limit,8192)]
                    if page or not data:
                        read.add(digest)
                        observed.setdefault(digest, []).append((offset, offset + len(page)))
                    observation = {'sha256': digest, 'offset': offset, 'size': len(data),
                                   'text': page.decode('utf-8', errors='replace'),
                                   'next_offset': offset + len(page) if offset + len(page) < len(data) else None}
                elif set(value) == {'knowledge_query'}:
                    payload = value['knowledge_query']
                    if self.query_knowledge is None:
                        raise ValueError('knowledge query is unavailable')
                    if not isinstance(payload, dict) or set(payload) != {'query'}:
                        raise ValueError('knowledge query requires query')
                    _text(payload['query'], 'query')
                    observation = self.query_knowledge(payload, remaining_seconds=remaining())
                    if not isinstance(observation, Mapping):
                        raise ValueError('knowledge result must be a mapping')
                    for memory in observation.get('memories', []):
                        versions.add((_text(memory['memory_id'], 'memory_id'),
                                      _text(memory['version'], 'version')))
                        for digest in memory.get('artifacts', {}).values():
                            known.add(_digest(digest))
                    citations.extend(observation.get('citations', []))
                    queried = True
                elif set(value) == {'request_check'}:
                    if self.request_check is None:
                        raise ValueError('additional checks are unavailable')
                    payload = value['request_check']
                    if not isinstance(payload, dict) or set(payload) != {'hypothesis', 'check'} or not isinstance(payload['check'], dict):
                        raise ValueError('check requires hypothesis and check object')
                    _text(payload['hypothesis'], 'hypothesis')
                    observation = self.request_check(payload, remaining_seconds=remaining())
                    if not isinstance(observation, Mapping) or type(observation.get('host_verified')) is not bool:
                        raise ValueError('check must return host_verified and artifact references')
                    host_ok = host_ok and observation['host_verified']
                    for digest in observation.get('artifacts', {}).values():
                        required.add(_digest(digest))
                        known.add(digest)
                else:
                    self._validate_decision(value, request, read, required, host_ok,
                                            queried, versions, citations, remaining, observed)
                    remaining()
                    self._record('decision', value)
                    return value
                remaining()
                self._record('tool', {'action': value, 'observation': observation})
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                observation = {'error': str(exc), 'expected_top_level': [
                    ['read_artifact'], ['knowledge_query'], ['request_check'],
                    ['decision', 'reason', 'evidence', 'repair', 'memory']],
                    'correction': 'Return only one of these exact key sets; remove API metadata and unknown keys.'}
                self._record('feedback', observation)
            messages.append({'role': 'user', 'content': json.dumps(observation, sort_keys=True)})

    def _validate_decision(self, value, request, read, required, host_ok, queried,
                           versions, citations, remaining, observed):
        if set(value) != {'decision', 'reason', 'evidence', 'repair', 'memory'} or value['decision'] not in {'accept', 'repair'}:
            raise ValueError('invalid terminal decision')
        _text(value['reason'], 'reason')
        self._evidence(value['evidence'], read)
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
        if memory['before'] is None:
            _text(memory['before_unavailable_reason'], 'before_unavailable_reason')
        elif memory['before_unavailable_reason'] is not None:
            raise ValueError('before snippet cannot have an unavailable reason')
        else:
            self._snippet(memory['before'], read, remaining, observed)
        self._snippet(memory['after'], read, remaining, observed)
        if memory['after']['sha256'] != request.source_sha256:
            raise ValueError('after snippet must reference reviewed current source')

    @staticmethod
    def _evidence(items, read):
        if not isinstance(items, list) or not items or any(not isinstance(item, str) or item not in read for item in items):
            raise ValueError('evidence must cite nonempty actually read artifact references')

    def _snippet(self, snippet, read, remaining, observed):
        if not isinstance(snippet, dict) or set(snippet) != {'sha256', 'start_line', 'end_line', 'text'}:
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
        if snippet['text'] != ''.join(lines[start - 1:end]):
            raise ValueError('snippet must match exact retained source lines')
        # A private full-artifact validation read is not an observation by the
        # agent. Quoted bytes must have appeared in one or more returned pages.
        quote_start = len(''.join(lines[:start - 1]).encode('utf-8'))
        quote_end = quote_start + len(snippet['text'].encode('utf-8'))
        covered = quote_start
        for page_start, page_end in sorted(observed.get(snippet['sha256'], [])):
            if page_start > covered:
                break
            covered = max(covered, page_end)
            if covered >= quote_end:
                return
        raise ValueError('read all quoted source bytes before quoting a snippet')
