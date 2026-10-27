"""One subject curriculum for learning, qualified targets and optimization."""
from dataclasses import asdict
import hashlib
import fcntl
import json
from pathlib import Path
import time

from optimizer.jobs import atomic, canonical
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .project_verifier import ProjectReview
from .subjects import SubjectScheduler, SubjectCurriculumAgent
from .subject_memory import SubjectMemory, StaleMemory
from .time_budget import AttemptTimeBudget


def _effective_project(projects, project_id):
    """Resolve a canceled wave entry to its current replacement evidence."""
    seen = set()
    while True:
        if project_id in seen:
            raise ValueError('cyclic project supersession')
        seen.add(project_id)
        record = projects.get(project_id, {})
        if not record.get('superseded_by'):
            return project_id, record
        project_id = record['superseded_by']


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

    def _decision_context(self):
        # Policy sees coverage and concise evidence, not repeated source/log bodies.
        from .subject_memory import digest
        accepted, counts = {}, {}
        for record in self.memory.records():
            subject = record['provenance'].get('subject_id', 'seed')
            counts[subject] = counts.get(subject, 0) + 1
            proposal = record['proposal']
            wave = record['provenance'].get('wave_index',0)
            if subject in accepted and accepted[subject]['wave_index'] > wave:
                continue
            accepted[subject] = dict(memory_id=record['memory_id'], version=digest(record),
                subject_id=subject, wave_index=wave, **{key:str(proposal[key])[:256] for key in
                    ('summary','applicability','preconditions','expectations','limitations')})
        reviews, rejected = {}, {}
        for key, row in self.state.get('memory_reviews', {}).items():
            subject = row.get('subject_id', 'seed')
            if row['outcome']=='rejected': rejected[subject]=rejected.get(subject,0)+1
            judgment = row.get('judgment') or {}
            if subject in reviews and reviews[subject][1].get('sequence',0) > row.get('sequence',0):
                continue
            reviews[subject] = (key, dict(row, judgment={field:(str(value)[:1024] if isinstance(value,str) else value)
                for field,value in judgment.items()}))
        projects = {}
        recent=sorted(self.state['projects'].items(),key=lambda pair:(pair[1].get('wave_index',0),pair[0]))[-4:]
        for key, row in recent:
            review = row.get('review') or {}
            projects[key] = dict(objective=row['objective'][:1024], review_sha256=row.get('review_sha256'),
                review={field:review.get(field) for field in ('decision','reason','evidence','repair')})
        return dict(accepted=list(accepted.values()), accepted_counts=counts,
            rejected_counts=rejected, memory_reviews=dict(reviews.values()), projects=projects,
            initial_memory=self.memory.initial[:2048],
            initial_memory_sha256=hashlib.sha256(self.memory.initial.encode()).hexdigest(),
            decision_error=self.state.get('decision_error'))

    def _settle_reviews(self):
        if self.memory_gate is None:
            return
        remaining = max(0, self.budget.deadline-self.clock())
        for result in self.memory_gate.poll(remaining_seconds=remaining):
            if result.get('manual_override'):
                self.scheduler.override_review(result['review_id'], result['manual_override'])
            else:
                self.scheduler.resolve_review(result['review_id'], result['outcome'])
            rows=self.state.setdefault('memory_reviews',{})
            if result['review_id'] not in rows or result.get('manual_override') and not rows[result['review_id']].get('manual_override'):
                prior = rows.get(result['review_id'])
                rows[result['review_id']] = dict(outcome=result['outcome'], judgment=result.get('judgment'),
                    rewrites=result.get('rewrites'), subject_id=result['provenance'].get('subject_id'),
                    sequence=max((row.get('sequence',0) for row in rows.values()),default=0)+1)
                if result.get('manual_override'):
                    rows[result['review_id']].update(manual_override=result['manual_override'], original_review=prior)
                if result['outcome']=='rejected':
                    self.state['needs_decision']=True
                if result.get('repair'):
                    provenance = result['provenance']
                    review = result['repair']
                    self.scheduler.add_repair(result['review_id']+'-memory', provenance['subject_id'],
                        provenance['project_id'], review['reason']+' '+json.dumps(review.get('repair'),sort_keys=True), review['evidence'])
                    self.state['needs_decision'] = True
                self._save()
            if hasattr(self.memory_gate, 'acknowledge'):
                self.memory_gate.acknowledge(result.get('delivery_id', result['review_id']))

    def _review(self, project_id, record):
        if record.get('review') is not None:
            return
        outcome = record['outcome']
        obligations = self.scheduler.state()['repairs']
        linked = {key: self.state['projects'].get(obligations.get(key, {}).get('project_id',key))
                  for key in record.get('repairs', [])}
        for key, original in linked.items():
            if original is not None and original.get('prior_project_sha256'):
                prior = self.artifacts.read_json(original['prior_project_sha256'])
                if (original is record or not original.get('review')
                        or prior.get('review_sha256') in obligations[key]['evidence']):
                    linked[key] = prior
        artifacts = dict(outcome['artifacts'])
        plan = record.get('plan')
        if plan is None:
            plan = next((item for item in (self.state.get('pending') or {}).get('items',[])
                         if item['project_id']==project_id), None)
        project = None
        if plan is not None:
            contract = self.artifacts.put_json(plan)
            artifacts['evidence:project-contract'] = contract
            project = dict(project_id=project_id, contract_sha256=contract,
                hypothesis=plan.get('spec',{}).get('hypothesis'), kind=plan.get('kind','graph'))
        for key, original in linked.items():
            if original is not None:
                artifacts.update({f'repair:{key}:{role}': sha for role, sha in original['outcome']['artifacts'].items()})
            artifacts.update({f'repair:{key}:evidence:{i}': sha
                for i,sha in enumerate(obligations.get(key,{}).get('evidence',[]))})
        sibling_evidence = {}
        pending = self.state.get('pending') or {}
        for pid in pending.get('projects', []):
            pid, sibling = _effective_project(self.state['projects'], pid)
            if (pid != project_id and pid not in record.get('supersedes', [])
                    and set(record.get('repairs', [])) & set(sibling.get('repairs', []))):
                sibling_evidence[pid] = sibling['outcome']['observations']
                artifacts.update({f'sibling:{pid}:{role}': sha for role, sha in sibling['outcome']['artifacts'].items()})
        objective = json.dumps(dict(subject=record['objective'], project=project, sibling_evidence=sibling_evidence, repairs={
            key: dict(original['review'], obligation=obligations.get(key))
            for key, original in linked.items() if original is not None}), sort_keys=True)
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
                      if not self.state['projects'][pid].get('superseded_by')
                      and self.state['projects'][pid]['review']['decision'] == 'accept'
                      and not self.state['projects'][pid].get('manual_review_id')
                      and self.state['projects'][pid]['review']['memory'] is not None]
            pending['review_ids'] = review_ids
            self._save()
        self.scheduler.complete_wave(pending['wave_id'], review_ids)
        subject = pending['subject']['subject_id']
        for pid in pending['projects']:
            record = self.state['projects'][pid]
            if record.get('superseded_by'):
                self.scheduler.supersede_review(pid, dict(record['cancellation'],
                    replacement_project=record['superseded_by']))
                continue
            review = record['review']
            if record.get('manual_review_id'):
                self.scheduler.supersede_review(pid, dict(request_id=record['manual_request'],
                    replacement_review=record['manual_review_id']))
                continue
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
            provenance = dict(project_id=pid, subject_id=subject, wave_index=len(self.state['waves'])+1,
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
        effective = dict(_effective_project(self.state['projects'], pid) for pid in pending['projects'])
        for record in effective.values():
            for repair_id in record.get('repairs', []):
                linked_repairs.setdefault(repair_id, []).append(record)
        obligations = self.scheduler.state()['repairs']
        for repair_id, records in linked_repairs.items():
            if (obligations[repair_id]['status'] == 'open'
                    and all(row['review']['decision'] == 'accept' for row in records)):
                self.scheduler.resolve_repair(repair_id, [row['review_sha256'] for row in records])
        if pending['wave_id'] not in self.state['waves']:
            self.state['waves'].append(pending['wave_id'])
        self.state.update(pending=None, needs_decision=True)
        self._save()

    def _run_wave(self, pending):
        for item in pending['items']:
            if (self.root / 'manual-pause.json').exists():
                raise TimeoutError('manual interaction queued at campaign checkpoint')
            self._remaining()
            pid = item['project_id']
            record = self.state['projects'].setdefault(pid, dict(
                objective=pending['subject']['objective'], repairs=item.get('repairs', []),
                wave_index=len(self.state['waves'])+1, plan=json.loads(canonical(item))))
            if record.get('superseded_by'):
                continue
            if 'outcome' not in record:
                directory = self.root / 'waves' / pending['wave_id'] / pid
                record['outcome'] = self.execute(item, self.artifacts.read_bytes(
                    pending['memory_sha256']), directory, remaining_seconds=self._remaining()).as_dict()
                self._save()
        # All linked probes and controls must finish before either is reviewed.
        for pid in pending['projects']:
            if (self.root / 'manual-pause.json').exists():
                raise TimeoutError('manual interaction queued at campaign checkpoint')
            if not self.state['projects'][pid].get('superseded_by'):
                self._review(pid, self.state['projects'][pid])
        self._publish_wave(pending)

    def _run(self):
        if self.memory_gate is not None and hasattr(self.memory_gate,'start_session'):
            self.memory_gate.start_session()
        self.budget = AttemptTimeBudget(self.root / 'elapsed.json', self.seconds, clock=self.clock)
        try:
            while True:
                self._remaining()
                if (self.root / 'manual-pause.json').exists():
                    raise TimeoutError('manual interaction queued at campaign checkpoint')
                self._settle_reviews()
                if self.state['pending'] is None and self.scheduler.prioritize_projects():
                    self.state.update(needs_decision=False, terminal=None)
                    self._save()
                if self.scheduler.is_complete() and not self.state['needs_decision']:
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
                        self.policy.decide(self.scheduler, memory_context=self._decision_context(),
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
                items = self.scheduler.claim_projects(wave_id)
                if not items:
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
            try:
                if self.memory_gate is not None:
                    self.memory_gate.close()
                    self._settle_reviews()
            finally:
                self.budget.close()

    def run(self):
        with (self.root / '.campaign.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('subject campaign is already active; observe its retained work') from None
            self._run()
            return self.report()

    resume = run
