"""Evidence-based learning hooks, with no retrieval or provider loop.

Single writer, as for CycleScheduler. load_project(row) is a trusted host adapter:
return project_id, outcome, evidence {CAS ID: {kind, summary}}, queries (verified
StrategyEvidence retrieval CAS IDs), usage (verified SkillUsage events), raw_refs.
It must resolve the owning run log and verify receipt scope and observable action
links; CAS existence alone is NOT admission. Copy cross-store artifacts first.
Evidence kinds: correctness, failure, recovery, avoidance, counterevidence, review, performance.
role_elapsed(operation_id) must read elapsed_seconds from the trusted host
role_finished ledger for that operation; elapsed_seconds excludes role setup.
Missing/malformed receipts suspend accounting and leave provider work pending.
Only this adapter can admit evidence; model proposals cannot introduce facts.

Callbacks match CycleScheduler. Validation receives immutable incumbent/candidate
bundle refs and frozen case IDs plus a durable operation ID. It is a host callback,
never a role. Unknown provider/validation work requires explicit reconciliation.
Raw instrumentation remains in CAS and its source logs for later SFT.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import time

from rsi.contracts import Mode
from rsi.cycles import SnapshotUpdate
from rsi.python_execution import atomic_json
from rsi.roles import AdmittedEvidence, _proposal
from rsi.skills import SkillBundle, SkillDocument


@dataclass(frozen=True)
class Validation:
    workflow: bool | None
    correctness: bool | None
    evidence_ref: str

    def __post_init__(self):
        if any(x is not None and type(x) is not bool for x in (self.workflow, self.correctness)):
            raise ValueError('validation gates require bool or unknown')
        if not self.evidence_ref:
            raise ValueError('host validation evidence required')


class Learning:
    def __init__(self, path, artifacts, memory, roles, *, load_project, validate,
                 mode='phase2', unused_window=10, context_chars=24000, clock=time.monotonic,
                 role_elapsed=None, request_seconds=600):
        if not callable(validate) or not callable(load_project):
            raise ValueError('host loader and validator required')
        if type(unused_window) is not int or unused_window <= 0 or context_chars < 1024:
            raise ValueError('invalid learning policy')
        if Mode(mode) != Mode.PHASE1 and not callable(role_elapsed):
            raise ValueError('required role_elapsed callback missing')
        if type(request_seconds) not in (int, float) or not math.isfinite(request_seconds) or request_seconds <= 0:
            raise ValueError('technical request timeout must be positive finite seconds')
        self.role_elapsed, self.request_seconds = role_elapsed, request_seconds
        self.path, self.store, self.memory, self.roles = Path(path), artifacts, memory, roles
        self.load_project, self.validate, self.clock = load_project, validate, clock
        self.mode, self.window, self.context_chars = Mode(mode), unused_window, context_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            active=None, versions=[], candidates={}, projects={}, operations={}, stopped=False)
        policy = dict(mode=self.mode.value, window=unused_window, context_chars=context_chars, request_seconds=request_seconds)
        if self.state.setdefault('policy', policy) != policy:
            raise ValueError('retained learning policy differs')
        self._save()

    @property
    def active(self):
        return self.state['active']

    def _save(self):
        atomic_json(self.path, self.state)

    def stop(self):
        self.state['stopped'] = True
        self._save()

    def resume(self):
        self.state['stopped'] = False
        self._save()

    def _begin(self, identity, request):
        existing = self.state['operations'].get(identity)
        if existing:
            if existing['request'] != request:
                raise ValueError('operation identity reused with different inputs')
            if 'result' not in existing:
                raise RuntimeError('unknown work: reconcile before retry')
            return existing['result']
        if self.state['stopped']:
            raise RuntimeError('learning stopped')
        self.state['operations'][identity] = dict(request=request)
        self._save()
        return None

    def _finish(self, identity, result):
        self.state['operations'][identity]['result'] = result
        self._save()
        return result

    def reconcile(self, operation_id, *, not_dispatched=False):
        """Clear only authoritative confirmed absence, never uncertain provider work.

        For completed writer work, retain its proposal via reconcile_proposal.
        Validation callbacks use the same idempotency ID to recover host receipts.
        """
        operation = self.state['operations'][operation_id]
        if 'result' in operation or not not_dispatched:
            raise RuntimeError('authoritative undispatched confirmation required')
        del self.state['operations'][operation_id]
        self._save()

    def _bundle(self, ref):
        bundle = SkillBundle.from_dict(self.store.read_json(ref))
        for doc in bundle.documents:
            import yaml  # Existing legacy dependency; no new parser or YAML subset.
            header = re.match(r'\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)', doc.content, re.DOTALL)
            try:
                metadata = yaml.safe_load(header[1]) if header else None
            except yaml.YAMLError as exc:
                raise ValueError('invalid skill metadata') from exc
            if (not isinstance(metadata, dict) or metadata.get('name') != doc.name
                    or not isinstance(metadata.get('description'), str)
                    or not metadata['description'].strip()):
                raise ValueError('invalid skill metadata')
        return bundle

    def _packet(self, row):
        packet = self.load_project(row)
        if packet['project_id'] != row['original_id'] or not packet['evidence']:
            raise ValueError('host evidence/project identity required')
        for ref, info in packet['evidence'].items():
            self.store.read_bytes(ref)
            if info['kind'] not in ('correctness', 'failure', 'recovery', 'avoidance', 'counterevidence', 'review', 'performance'):
                raise ValueError('unsupported evidence kind')
        for ref in packet['queries']:
            if self.store.read_json(ref)['kind'] != 'retrieval':
                raise ValueError('verified query receipt required')
        for ref in packet['raw_refs']:
            self.store.read_bytes(ref)
        return json.loads(json.dumps(packet, allow_nan=False))

    def memory_update(self, baseline, results, ctx):
        identity = ctx['operation_id']
        request = dict(kind='memory', baseline=baseline, results=results, recovery=ctx.get('recovery', False))
        cached = self._begin(identity, request)
        if cached is not None:
            return SnapshotUpdate(**cached)
        self.store.read_bytes(baseline)
        for row in results:
            packet = self._packet(row)
            project = packet['project_id']
            packet_ref = self.store.put_json(packet)
            # Outcome records describe observations, never infer reusable correctness.
            kinds = {v['kind'] for v in packet['evidence'].values()}
            kind = 'correctness' if packet['outcome'] == 'accepted' and 'correctness' in kinds else 'failure'
            proposal = dict(base_versions=[], kind=kind, project_id=project,
                            summary=packet['outcome'], evidence=list(packet['evidence']))
            self.memory.accept(identity + ':' + project, proposal,
                dict(packet=packet_ref, baseline=baseline, queries=packet['queries'], raw_refs=packet['raw_refs']))
            if not ctx.get('recovery'):
                # Unique originals, not calls or recoveries, determine exposure age.
                self.state['projects'].setdefault(project, packet_ref)
        reference = self.store.put_bytes(self.memory.snapshot())
        result = dict(reference=reference, actor_seconds=0.0)
        self._finish(identity, result)
        return SnapshotUpdate(**result)

    def removal_eligible(self, skill, section, *, safety=False, counterevidence=False, reviewed=False):
        exposed, used = set(), set()
        for project, ref in self.state['projects'].items():
            for event in self.store.read_json(ref)['usage']:
                if event['skill'] == skill and event.get('section') in (None, section):
                    if event['kind'] == 'exposed':
                        exposed.add(project)
                    if event['kind'] in ('cited', 'applied'):
                        used.add(project)
        return (len(exposed) >= self.window and not used
                and (not safety or counterevidence or reviewed))

    def aggregate(self, current=None):
        """Summarise unique originals, preserving raw references and negative rows.

        Host-admitted outcomes are kept next to retrieval frequency; the latter
        never receives an automatic usefulness score. Priority orders failure
        avoidance/detection/recovery before correctness then performance ideas.
        """
        packets = {key: self.store.read_json(ref) for key, ref in self.state['projects'].items()}
        if current is not None:
            packets[current['project_id']] = current
        queries, sections, outcomes = {}, {}, []
        priority = {'counterevidence': 0, 'failure': 1, 'avoidance': 1, 'recovery': 2,
                    'correctness': 3, 'review': 3, 'performance': 4}
        for project, packet in packets.items():
            for ref, info in packet['evidence'].items():
                outcomes.append(dict(project=project, reference=ref, kind=info['kind'],
                                     summary=info['summary'][:500]))
            for reference in packet['queries']:
                response = self.store.read_json(reference)['payload']['response']
                queries[reference] = dict(project=project,
                    results=[json.dumps(row, sort_keys=True)[:500] for row in response['results'][:5]],
                    negative_records=[json.dumps(row, sort_keys=True)[:500]
                                      for row in response['negative_records'][:5]],
                    evidence_ids=list(packet['evidence']))
            for event in packet['usage']:
                key = event['skill'] + ':' + str(event.get('section'))
                counts = sections.setdefault(key, {})
                counts.setdefault(event['kind'], set()).add(project)
        return dict(outcomes=sorted(outcomes, key=lambda x: priority[x['kind']]),
                    queries=queries, sections={key: {kind: sorted(ids) for kind, ids in counts.items()}
                                              for key, counts in sections.items()})

    def skill_update(self, incumbent, memory_ref, row, ctx):
        if self.mode == Mode.PHASE1:
            return SnapshotUpdate(incumbent)
        identity = ctx['operation_id']
        request = dict(kind='skill', incumbent=incumbent, memory=memory_ref, row=row)
        cached = self._begin(identity, request)
        if cached is not None:
            return SnapshotUpdate(**cached)
        bundle = self._bundle(incumbent)
        self.store.read_bytes(memory_ref)
        if self.active is None:
            self.state['active'] = incumbent
        if incumbent not in self.state['versions']:
            self.state['versions'].append(incumbent)
        packet = self._packet(row)
        packet_ref = self.store.put_json(packet)
        self.state['operations'][identity]['packet'] = packet_ref
        self._save()  # Freeze evidence and incumbent before invoking the provider.
        evidence = [AdmittedEvidence(ref, dict(kind=info['kind'], summary=info['summary'][:1000]))
                    for ref, info in packet['evidence'].items()]
        # Bounded context; never truncate identifiers, baseline or raw retained CAS.
        prompt = ('Prioritise error avoidance, detection and recovery before optimisation. '
            'Retrieval frequency is not usefulness. Negative results and rejected negative ideas '
            'can demonstrate valuable avoidance. Cite admitted outcome evidence. '
            'replace sets only Learned guidance; remove deletes one exact learned fragment, '
            'Add the section when absent. A complete YAML-frontmatter document replacement, '
            'or exact removal outside Learned guidance, requires cited host review/counterevidence. '
            'Never delete a mapped skill. Preserve safety/recovery unless '
            'counterevidence or explicit review supports removal.\n')
        context = dict(bundle=incumbent, memory=memory_ref, packet=packet_ref,
                       skills={d.name: d.read('learned-guidance') if 'learned-guidance' in d.sections else ''
                               for d in bundle.documents}, aggregate=self.aggregate(packet))
        # Retain full aggregates in CAS; bound only model context to recent projects.
        context['aggregate_ref'] = self.store.put_json(context['aggregate'])
        context['aggregate']['queries'] = dict(list(context['aggregate']['queries'].items())[-10:])
        context['aggregate']['outcomes'] = context['aggregate']['outcomes'][:10]
        context['aggregate']['sections'] = {key: {kind: ids[-10:] for kind, ids in counts.items()}
                                            for key, counts in context['aggregate']['sections'].items()}
        encoded = json.dumps(context, sort_keys=True)
        if len(encoded) + len(prompt) + len(json.dumps([asdict(e) for e in evidence])) > self.context_chars:
            raise ValueError('learning context exceeds bound; host must summarise admitted inputs')
        remaining = ctx['remaining']['actor_seconds']
        allowance = self.request_seconds if remaining is None else min(remaining, self.request_seconds)
        if allowance <= 0:
            raise TimeoutError('actor allocation exhausted')
        proposal = self.roles.skillwriter(prompt + encoded, evidence,
                                          deadline=self.clock() + allowance)
        proposal_ref = self.store.put_json(asdict(proposal))
        operation = self.state['operations'][identity]
        operation['proposal'] = proposal_ref
        self._save()  # Retain provider output before reading host accounting.
        try:
            elapsed = self.role_elapsed(identity)
        except Exception:
            elapsed = None
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            operation['accounting_incomplete'] = True
            self._save()
            raise RuntimeError('accounting incomplete: role_finished receipt required; reconcile')
        operation['role_finished'] = dict(operation_id=identity, elapsed_seconds=elapsed)
        self._save()
        if elapsed > allowance:
            raise TimeoutError('measured actor allocation exceeded; reconcile retained receipt')
        return self.reconcile_proposal(identity, proposal_ref, elapsed)

    def reconcile_proposal(self, identity, proposal_ref, actor_seconds):
        SnapshotUpdate(proposal_ref, actor_seconds)  # validates elapsed accounting
        operation = self.state['operations'][identity]
        if 'result' in operation:
            return SnapshotUpdate(**operation['result'])
        request = operation['request']
        if request['kind'] != 'skill':
            raise ValueError('not a writer operation')
        incumbent = request['incumbent']
        bundle = self._bundle(incumbent)
        packet = self.store.read_json(operation['packet'])
        proposal = _proposal('skillwriter', self.store.read_json(proposal_ref), set(packet['evidence']))
        if not proposal.evidence_ids:
            raise ValueError('proposal requires admitted outcome support')
        docs = {d.name: d for d in bundle.documents}
        protected = any(packet['evidence'][ref]['kind'] in ('counterevidence', 'review')
                        for ref in proposal.evidence_ids)
        for edit in proposal.edits:
            if edit.skill not in docs or not edit.content.strip():
                raise ValueError('known skill and nonempty learned fragment required')
            doc = docs[edit.skill]
            learned = 'learned-guidance' in doc.sections
            section = doc.read('learned-guidance') if learned else ''
            full_document = edit.operation == 'replace' and edit.content.startswith('---')
            outside = edit.operation == 'remove' and edit.content not in section
            if full_document or outside:
                if not protected:
                    raise ValueError('baseline correction requires host review/counterevidence')
                if full_document:
                    content = edit.content
                else:
                    if doc.content.count(edit.content) != 1:
                        raise ValueError('remove must identify one exact baseline fragment')
                    content = doc.content.replace(edit.content, '', 1)
                docs[edit.skill] = SkillDocument(edit.skill, content)
                continue
            if edit.operation == 'remove':
                if not self.removal_eligible(edit.skill, 'learned-guidance', safety=True,
                                             counterevidence=protected):
                    raise ValueError('removal lacks exposure window or safety review')
                if section.count(edit.content) != 1:
                    raise ValueError('remove must identify one exact learned fragment')
                replacement = section.replace(edit.content, '', 1)
            else:
                replacement = edit.content + '\n'
            if re.search(r'^#{1,2} |^---$', replacement, re.MULTILINE):
                raise ValueError('learned fragment cannot change document structure')
            marker = '## Learned guidance\n'
            if learned:
                # Resolve by parsed section content, independent of heading spelling.
                heading = next(m for m in re.finditer(r'^## ([^\n]+)\n', doc.content, re.MULTILINE)
                    if re.sub(r'[^a-z0-9]+', '-', m[1].lower()).strip('-') == 'learned-guidance')
                content = doc.content[:heading.end()] + replacement + doc.content[heading.end() + len(section):]
            else:
                content = doc.content + ('' if doc.content.endswith('\n') else '\n') + marker + replacement
            docs[edit.skill] = SkillDocument(edit.skill, content)
        candidate = SkillBundle(tuple(docs.values()), bundle.mapping)
        reference = self.store.put_json(candidate.to_dict())
        self._bundle(reference)
        self.state['candidates'][reference] = dict(incumbent=incumbent, proposal=proposal_ref,
            packet=self.store.put_json(packet), memory=request['memory'], quarantined=False)
        return SnapshotUpdate(**self._finish(identity, dict(reference=reference, actor_seconds=actor_seconds)))

    def evaluate(self, candidate, cases, operation_id):
        cases = tuple(cases)
        if not cases:
            raise ValueError('frozen validation cases required')
        record = self.state['candidates'][candidate]
        request = dict(kind='validation', candidate=candidate, incumbent=record['incumbent'], cases=list(cases))
        cached = self._begin(operation_id, request)
        if cached is not None:
            return cached['active']
        verdict = self.validate(record['incumbent'], candidate, cases, operation_id)
        return self.reconcile_validation(operation_id, verdict)

    def reconcile_validation(self, operation_id, verdict):
        operation = self.state['operations'][operation_id]
        if 'result' in operation:
            return operation['result']['active']
        if not isinstance(verdict, Validation) or operation['request']['kind'] != 'validation':
            raise ValueError('host validation receipt required')
        request = operation['request']
        self.store.read_bytes(verdict.evidence_ref)
        record = self.state['candidates'][request['candidate']]
        record['validation'] = dict(asdict(verdict), cases=request['cases'])
        if (verdict.workflow is True and verdict.correctness is True and not record['quarantined']
                and self.active == request['incumbent']):
            self.state['active'] = request['candidate']
            if self.active not in self.state['versions']:
                self.state['versions'].append(self.active)
        self._finish(operation_id, dict(active=self.active))
        return self.active

    def quarantine(self, candidate, counterevidence_ref):
        record = self.state['candidates'][candidate]
        packet = self.store.read_json(record['packet'])
        if packet['evidence'].get(counterevidence_ref, {}).get('kind') != 'counterevidence':
            raise ValueError('host-admitted counterevidence required')
        record['quarantined'] = counterevidence_ref
        if self.active == candidate:
            self.state['active'] = record['incumbent']
        self._save()

    def rollback(self, reference):
        if reference not in self.state['versions']:
            raise ValueError('rollback requires previously active immutable version')
        self._bundle(reference)
        self.state['active'] = reference
        self._save()
