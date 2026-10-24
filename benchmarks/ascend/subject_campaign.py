"""One subject curriculum for learning, qualified targets and optimization."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .project_verifier import ProjectReview
from .subjects import SubjectScheduler, SubjectCurriculumAgent
from .subject_memory import SubjectMemory, StaleMemory
from .time_budget import AttemptTimeBudget


class SubjectCampaign:
    def __init__(self, root, subjects, initial_memory, *, select_wave, execute, verifier,
                 complete, max_seconds=7200, memory_gate=None, index_memory=None,
                 clock=time.monotonic):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.scheduler = SubjectScheduler(self.root, subjects)
        self.memory = SubjectMemory(self.root / 'memory', initial_memory, index=index_memory)
        self.artifacts = ContentAddressedArtifacts(self.root / 'artifacts')
        self.select_wave, self.execute, self.verifier = select_wave, execute, verifier
        self.policy = SubjectCurriculumAgent(complete)
        self.seconds, self.clock, self.memory_gate = max_seconds, clock, memory_gate
        self.path = self.root / 'subject-campaign.json'
        self.state = json.loads(self.path.read_bytes()) if self.path.exists() else dict(
            schema='ascend-subject-campaign-v1', waves=[], projects={}, pending=None,
            needs_decision=False, terminal=None)

    def _save(self):
        atomic(self.path, canonical(self.state))

    def _remaining(self):
        available = self.budget.deadline - self.clock()
        if available <= 0:
            raise TimeoutError('subject campaign elapsed allocation exhausted')
        return available

    def report(self):
        return dict(self.state, subjects=self.scheduler.context(), accepted_memory=self.memory.records())

    def _settle_reviews(self):
        if self.memory_gate is None:
            return
        for result in self.memory_gate.poll(remaining_seconds=self._remaining()):
            self.scheduler.resolve_review(result['review_id'], result['outcome'])
            if result.get('repair'):
                provenance = result['provenance']
                review = result['repair']
                self.scheduler.add_repair(result['review_id']+'-memory', provenance['subject_id'],
                    provenance['project_id'], review['reason'], review['evidence'])
                self.state['needs_decision'] = True
                self._save()
            if hasattr(self.memory_gate, 'acknowledge'):
                self.memory_gate.acknowledge(result['review_id'])

    def _review(self, project_id, record):
        if record.get('review') is not None:
            return
        outcome = record['outcome']
        linked = {key: self.state['projects'].get(key) for key in record.get('repairs', [])}
        artifacts = dict(outcome['artifacts'])
        for key, original in linked.items():
            if original is not None:
                artifacts.update({f'repair:{key}:{role}': sha for role, sha in original['outcome']['artifacts'].items()})
        sibling_evidence = {}
        pending = self.state.get('pending') or {}
        for pid in pending.get('projects', []):
            sibling = self.state['projects'].get(pid, {})
            if pid != project_id and set(record.get('repairs', [])) & set(sibling.get('repairs', [])):
                sibling_evidence[pid] = sibling['outcome']['observations']
                artifacts.update({f'sibling:{pid}:{role}': sha for role, sha in sibling['outcome']['artifacts'].items()})
        objective = json.dumps(dict(subject=record['objective'], sibling_evidence=sibling_evidence, repairs={
            key: original['review'] for key, original in linked.items() if original is not None}), sort_keys=True)
        request = ProjectReview(project_id=project_id, source_sha256=outcome['source_sha256'],
            host_verified=outcome['host_verified'], artifacts=artifacts,
            host_facts=tuple(outcome['observations'].get('host_facts', ())),
            objective=objective)
        try:
            review = self.verifier.review(request, max_seconds=self._remaining(),
                feedback=record.get('rewrite_feedback'))
        finally:
            record['verifier_trace'] = self.artifacts.put_json(self.verifier.last_trace)
            self._save()
        record.pop('rewrite_feedback', None)
        record['review'] = review
        record['review_request'] = asdict(request)
        record['review_sha256'] = self.artifacts.put_json(review)
        self._save()

    def _publish_wave(self, pending):
        review_ids = pending.get('review_ids')
        if review_ids is None:
            review_ids = [pid for pid in pending['projects']
                      if self.state['projects'][pid]['review']['decision'] == 'accept'
                      and self.state['projects'][pid]['review']['memory'] is not None]
            pending['review_ids'] = review_ids
            self._save()
        self.scheduler.complete_wave(pending['wave_id'], review_ids)
        subject = pending['subject']['subject_id']
        for pid in pending['projects']:
            record = self.state['projects'][pid]
            review = record['review']
            evidence = [record['review_sha256']]
            if review['decision'] == 'repair':
                if pid in review_ids:
                    self.scheduler.resolve_review(pid, 'rejected')
                self.scheduler.add_repair(pid, subject, pid, review['reason'], evidence)
                continue
            if review['memory'] is None:
                if pid in review_ids:
                    self.scheduler.resolve_review(pid, 'accepted')
                continue
            provenance = dict(project_id=pid, subject_id=subject,
                review_sha256=record['review_sha256'], request=record['review_request'])
            if self.memory_gate is None:
                try:
                    self.memory.accept(pid, review['memory'], provenance)
                except StaleMemory as error:
                    record.setdefault('prior_reviews', []).append(review)
                    record['rewrite_feedback'] = dict(reason=str(error), previous_proposal=review['memory'],
                        current_memories=self.memory.records())
                    record['review'] = None
                    self._save()
                    self._review(pid, record)
                    return self._publish_wave(pending)
                self.scheduler.resolve_review(pid, 'accepted')
            else:
                self.memory_gate.submit(pid, review['memory'], provenance,
                    record['review_request'], remaining_seconds=self._remaining())
        linked_repairs = {}
        for pid in pending['projects']:
            for repair_id in self.state['projects'][pid].get('repairs', []):
                linked_repairs.setdefault(repair_id, []).append(self.state['projects'][pid])
        for repair_id, records in linked_repairs.items():
            if all(row['review']['decision'] == 'accept' for row in records):
                self.scheduler.resolve_repair(repair_id, [row['review_sha256'] for row in records])
        if pending['wave_id'] not in self.state['waves']:
            self.state['waves'].append(pending['wave_id'])
        self.state.update(pending=None, needs_decision=True)
        self._save()

    def _run_wave(self, pending):
        for item in pending['items']:
            self._remaining()
            pid = item['project_id']
            record = self.state['projects'].setdefault(pid, dict(
                objective=pending['subject']['objective'], repairs=item.get('repairs', [])))
            if 'outcome' not in record:
                directory = self.root / 'waves' / pending['wave_id'] / pid
                record['outcome'] = self.execute(item, self.artifacts.read_bytes(
                    pending['memory_sha256']), directory, remaining_seconds=self._remaining()).as_dict()
                self._save()
        # All linked probes and controls must finish before either is reviewed.
        for pid in pending['projects']:
            self._review(pid, self.state['projects'][pid])
        self._publish_wave(pending)

    def run(self):
        self.budget = AttemptTimeBudget(self.root / 'elapsed.json', self.seconds, clock=self.clock)
        try:
            while True:
                self._remaining()
                self._settle_reviews()
                if self.scheduler.is_complete():
                    self.state['terminal'] = 'COMPLETE'
                    self._save()
                    return self.report()
                if self.state['pending'] is not None:
                    self.scheduler.start_wave(self.state['pending']['wave_id'], self.state['pending']['memory_sha256'])
                    self._run_wave(self.state['pending'])
                    continue
                if self.state['needs_decision']:
                    # Invalid provider decisions get repair feedback on the next call,
                    # within the same elapsed allocation, never an attempt-count limit.
                    try:
                        self.policy.decide(self.scheduler, memory_context=dict(
                            accepted=self.memory.records(), decision_error=self.state.get('decision_error'),
                            projects={key: {field: record.get(field) for field in ('objective', 'review', 'review_sha256')}
                                      for key, record in list(self.state['projects'].items())[-8:]}),
                            remaining_seconds=self._remaining())
                    except (ValueError, KeyError) as error:
                        self.state['decision_error'] = str(error)
                        self._save()
                        continue
                    self.state.pop('decision_error', None)
                    self.state.update(needs_decision=False)
                    self._save()
                    continue
                context = self.scheduler.context()
                if context['current_subject'] is None:
                    if self.memory_gate is None:
                        raise RuntimeError('unfinished obligations require a curriculum revisit')
                    self.memory_gate.wait(remaining_seconds=self._remaining())
                    self.state['needs_decision'] = bool(self.scheduler.context()['repairs'])
                    continue
                snapshot = self.memory.snapshot()
                wave_id = f"wave-{len(self.state['waves']) + 1:05d}"
                items = self.select_wave(context, snapshot, self.state['projects'],
                                         remaining_seconds=self._remaining())
                if not items or len({i['project_id'] for i in items}) != len(items):
                    raise ValueError('wave must contain unique projects')
                if set(i['project_id'] for i in items) & self.state['projects'].keys():
                    raise ValueError('project identity already executed')
                memory_sha = self.artifacts.put_bytes(snapshot)
                self.state['pending'] = dict(wave_id=wave_id, subject=context['current_subject'],
                    memory_sha256=memory_sha, items=items, projects=[i['project_id'] for i in items])
                self._save()
        except TimeoutError:
            self.state['terminal'] = ('TIME_BUDGET_EXHAUSTED' if self.clock() >= self.budget.deadline
                                      else 'REVIEW_PENDING')
            self._save()
            return self.report()
        finally:
            self.budget.close()

    resume = run
