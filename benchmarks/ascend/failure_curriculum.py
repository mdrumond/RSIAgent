"""Explicit Phase 2 observations guiding small Phase 1 experiments, not facts."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from optimizer.jobs import JobStore, atomic, canonical, sha
from .phase2_feedback import FeedbackStore


GUIDANCE = '''Investigate the supplied Phase 1, Phase 2 and optimizer observations, which are failures or
working controls, NOT verified reusable knowledge. Distinguish learnable kernel,
numerical, compilation and performance issues from infrastructure/provider errors,
invalid specifications and suspected unsupported/impossible requirements. A budget
exhaustion never proves impossibility. Explain evidence-based deferrals; do not
invent projects to repair infrastructure. Read retained source/host diagnostics
with {"read_artifact":{"sha256":"exact hash","offset":0,"limit":8192}};
at most six reads per selection. Failure cases include exact input/expected/observed
artifact references and availability explanations; read these before selecting a repair. Raw profiling stays remote; use compact digests.
The selection-index lists EVERY selected failure, including observations outside
the current detailed page. Use its classifications/summaries to defer infrastructure
or provider failures without inventing kernel work; its hashes are also readable.
Propose small experiments isolating one hypothesis, a working control, and an
explicit observable result. Sibling projects share pre-wave memory; they do NOT
depend on each other. Existing verified projects may serve as controls.
Add optional top-level failure_investigations to the ordinary DECISION JSON,
outside the strict WaveSpec: [{"feedback_id":"...","status":"investigating",
"classification":"learnable","reason":"...","project_id":"probe-id",
"control_project_id":"control-id","hypothesis":"...","observation":"..."}].
Prioritize remediation; fill remaining wave slots with new projects. Only remedial
projects require probe/control lineage in this list. Use failure_mode=none and omit
failure-repair for exploratory probes doing a different computation. A repair
requires the attached observed failing case for the same graph, dtype, exact
inputs, expected results and comparison policy; never ask the actor to invent
or rediscover a mismatch. The observer supplies the failing example.
Classifications: learnable, infrastructure, provider, invalid-specification,
suspected-unsupported, suspected-impossible. Status: investigating, deferred,
ready-to-retry. Deferred needs reason; budget exhaustion alone is not a reason
to label a specification impossible. Recommend ready-to-retry only for a previously
linked, committed experiment with retained host evidence and a verified control;
the probe may have failed usefully. This recommends a retry, never proves a root
cause or resolves the original task. Terminal decisions may carry the same list.
The goal of remediation is to improve outcomes on the original failed project.
Once a linked experiment has committed host evidence and a verified control,
prefer recommending ready-to-retry over generating further probes when that
evidence suggests a plausible repair. The original retry is itself an experiment:
it tests whether the learned knowledge transfers. You do not need a complete
root-cause explanation before recommending it. Further investigation may be
appropriate. If you keep a failure investigating despite eligible probe/control
evidence, use the existing reason field to identify the remaining uncertainty and
explain why another probe is more useful than retrying the original now. Consider
the remaining elapsed allowance and leave time for the original retry. A passing
remedial project alone never proves the original is fixed.
Never mark a failure resolved: only a successful original-target retry can do so.'''


def select_feedback(root, target, ids=(), *, retry_id=None):
    if not Path(root).is_dir():
        raise ValueError('feedback root does not exist')
    records = FeedbackStore(root).records()
    by_id = {row['feedback_id']: row for row in records}
    language = 'ascend-c' if target == 'a3' else 'catlass-dsl'
    def compatible(outcome):
        normalized = {'native-a3': 'a3', 'native-a5': 'a5'}.get(outcome['target'], outcome['target'])
        return normalized == target and outcome['language'] == language
    selected = list(ids) if ids else [row['feedback_id'] for row in records
        if row['investigation']['status'] in {'open', 'investigating', 'ready-to-retry'}
        and compatible(row['outcome'])]
    if retry_id:
        selected = [retry_id]
    if not selected:
        raise ValueError('no unresolved failure feedback selected')
    result = []
    for key in dict.fromkeys(selected):
        if key not in by_id:
            raise ValueError('unknown failure feedback ID: ' + key)
        row = by_id[key]
        outcome = row['outcome']
        if not compatible(outcome):
            raise ValueError('feedback target/language incompatible with selected campaign')
        result.append(row)
    # Successful siblings are context, never additional investigation obligations.
    controls = [row for row in records if row['outcome']['verified'] and
        compatible(row['outcome']) and
        row['feedback_id'] not in selected][:8]
    return dict(selected=result, controls=controls)


class FailureCurriculum:
    def __init__(self, root, store, selection):
        self.root, self.store, self.selection = Path(root), FeedbackStore(store), selection
        self.path = self.root / 'failure-investigations.json'
        self.lineage = json.loads(self.path.read_bytes()) if self.path.exists() else []
        self.ids = {r['feedback_id'] for r in selection['selected']}

    def refresh(self, target):
        """Reload unresolved observations before selecting the next mixed wave."""
        try:
            self.selection = select_feedback(self.store.root, target)
        except ValueError as error:
            if str(error) != 'no unresolved failure feedback selected':
                raise
            self.selection = dict(selected=[], controls=[])
        self.ids = {row['feedback_id'] for row in self.selection['selected']}
        return self.selection

    def context(self, state):
        self.state = state
        rows = []
        for role in ('selected', 'controls'):
            for row in self.selection[role]:
                out = row['outcome']
                rows.append(dict(feedback_id=row['feedback_id'], role=role,
                    category=out['category'], lifecycle=row.get('lifecycle'),
                    retries=row['investigation']['retries'], task={k: v for k, v in out['task'].items() if k != 'submission'},
                    task_sha256=sha(canonical(out['task'])),
                    submission_preview=json.dumps(out['task'].get('submission'), sort_keys=True)[:2048],
                    investigations=[r for r in self.lineage if r['feedback_id'] == row['feedback_id']][-3:],
                    interpretation=out['interpretation'],
                    host_result=json.dumps(out['result'].get('result'), sort_keys=True)[:4096],
                    speedup=out['result'].get('speedup'), evidence=out['evidence'],
                    failure_cases=self.store.failure_cases(row['feedback_id'])))
        # Keep source/evidence hash-readable, never put entire candidate/report in context.
        from .agent_presentation import _preview
        start = ((state.get('wave_index', 1)-1) * 8) % max(1, len(self.selection['selected']))
        chosen = [r for r in rows if r['role'] == 'selected'][start:start+8] + [r for r in rows if r['role'] == 'controls'][:2]
        manifest = []
        latest = {row['feedback_id']: row for row in self.lineage}
        for row in self.selection['selected']:
            out = row['outcome']
            investigation = latest.get(row['feedback_id'], row['investigation'])
            manifest.append(dict(feedback_id=row['feedback_id'], category=out['category'],
                status=investigation['status'], classification=investigation.get('classification') or out['result'].get('classification'),
                reason=(investigation.get('reason') or out['result'].get('summary') or '')[:256],
                task_sha256=sha(canonical(out['task'])),
                evidence={role: ref['sha256'] for role, ref in out['evidence'].items()}))
        return (*(_preview(row) for row in chosen), {'role': 'selection-index', 'selected': manifest})

    def read(self, sha256, offset=0, limit=8192):
        digest = sha256
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8192:
            raise ValueError('invalid feedback artifact page')
        for row in self.selection['selected'] + self.selection['controls']:
            task = canonical(row['outcome']['task'])
            if sha(task) == digest:
                return dict(sha256=digest, offset=offset, size=len(task),
                    text=task[offset:offset+limit].decode(errors='replace'),
                    next_offset=offset+limit if offset+limit < len(task) else None)
            try:
                data = self.store.failure_artifact_bytes(row['feedback_id'], digest)
            except (ValueError, FileNotFoundError):
                pass
            else:
                return dict(sha256=digest, offset=offset, size=len(data),
                    text=data[offset:offset+limit].decode(errors='replace'),
                    next_offset=offset+limit if offset+limit < len(data) else None)
            for ref in row['outcome']['evidence'].values():
                if ref['sha256'] == digest:
                    return self.store.read_evidence(digest, offset=offset, limit=limit)
        raise ValueError('artifact is not in selected feedback')

    def admit(self, rows, wave, context):
        if not isinstance(rows, list):
            raise ValueError('failure_investigations must be a list')
        projects = {p.spec.project_id for p in wave.projects} if wave else set()
        prior = {p.project_id for p in context.prior_outcomes if p.verified}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('failure_investigations entries must be objects')
            if row.get('feedback_id') not in self.ids or row.get('status') not in {
                    'investigating', 'ready-to-retry', 'deferred'}:
                raise ValueError('unknown feedback or investigation status')
            if row.get('classification') not in {'learnable', 'infrastructure', 'provider',
                    'invalid-specification', 'suspected-unsupported', 'suspected-impossible'}:
                raise ValueError('unknown failure classification')
            if not isinstance(row.get('reason'), str) or not row['reason'].strip():
                raise ValueError('investigation requires reason')
            if row['status'] == 'deferred':
                continue
            probe, control = row.get('project_id'), row.get('control_project_id')
            if not probe or probe == control or control not in projects | prior:
                raise ValueError('investigation requires a distinct working control')
            if row['status'] == 'investigating':
                if probe not in projects or not row.get('hypothesis') or not row.get('observation'):
                    raise ValueError('investigation requires a small probe, hypothesis and observation')
            elif not any(old['feedback_id'] == row['feedback_id'] and
                    old.get('project_id') == probe and old.get('control_project_id') == control
                    for old in self.lineage):
                raise ValueError('retry recommendation requires a previously linked probe/control')
            elif not self._supported(row, self.state):
                raise ValueError('retry recommendation requires committed host probe/control evidence')
        latest = {row['feedback_id']: row for row in [*self.lineage, *rows]}
        if wave is None and any(latest.get(key, {}).get('status') not in
                {'ready-to-retry', 'deferred'} for key in self.ids):
            raise ValueError('terminal feedback curriculum must recommend retry or explain deferral for every selected failure')
        if wave is not None:
            for project in wave.projects:
                spec = project.spec
                keys = [row['feedback_id'] for row in rows if row.get('project_id') == spec.project_id]
                mode = getattr(spec, 'failure_mode', None)
                if mode is not None and mode.value != 'none':
                    from .failure_handoff import FailureContext
                    FailureContext(dict(root=str(self.store.root), feedback_ids=keys), spec).require_repair_evidence(spec)
        return tuple(rows)

    def project_context(self, project_id):
        keys = list(dict.fromkeys(row['feedback_id'] for row in self.lineage
                                 if row.get('project_id') == project_id and row['feedback_id'] in self.ids))
        return dict(root=str(self.store.root), feedback_ids=keys) if keys else None

    def record(self, rows, state=None):
        if state is not None:
            self.state = state
        if rows and self.lineage[-len(rows):] != list(rows):
            for row in rows:
                latest = next((old for old in reversed(self.lineage)
                               if old['feedback_id'] == row['feedback_id']), None)
                if row != latest:
                    self.lineage.append(row)
        atomic(self.path, canonical(self.lineage))
        if hasattr(self, 'state'):
            self.sync(self.state)

    @staticmethod
    def _supported(row, state):
        memories = {r['project_id']: json.loads(base64.b64decode(raw)) for r, raw in zip(
            state.get('prior_outcomes', []), state.get('memory_bytes_base64', []), strict=True)}
        failed = {r['project_id'] for r in state.get('failed_outcomes', [])
            if r.get('evidence', {}).get('sha256') and r['evidence'].get('source_sha256') and (
                r['evidence'].get('classification') in {'host-mismatch', 'compile-failure'} or
                any(value > 0 for name, value in (r['evidence'].get('action_counts') or {}).items()
                    if name in {'compile', 'run', 'request-compile', 'request-run'}))}
        probe, control = row.get('project_id'), row.get('control_project_id')
        return (bool(memories.get(control, {}).get('host_facts')) and
                (bool(memories.get(probe, {}).get('host_facts')) or probe in failed))

    def sync(self, state):
        failed = {r['project_id']: r for r in state.get('failed_outcomes', [])}
        memories = {row['project_id']: json.loads(base64.b64decode(encoded)) for row, encoded in
            zip(state.get('prior_outcomes', []), state.get('memory_bytes_base64', []), strict=True)}
        for key in self.ids:
            rows = [row for row in self.lineage if row['feedback_id'] == key]
            if not rows:
                continue
            latest = rows[-1]
            evidence = []
            for row in rows:
                for pid in (row.get('project_id'), row.get('control_project_id')):
                    memory = memories.get(pid, {})
                    evidence += [dict(project_id=pid, source_sha256=memory['source_revision'],
                        evidence_sha256=fact['evidence_sha256']) for fact in memory.get('host_facts', [])]
                    if pid in failed:
                        evidence.append(dict(project_id=pid, failed_observation=failed[pid]))
            status = latest['status']
            if status == 'ready-to-retry' and not self._supported(latest, state):
                raise ValueError('retry recommendation requires committed host probe/control evidence')
            def merge(current):
                if current['status'] == 'resolved':
                    return None
                other = [r for r in current['projects'] if
                         not isinstance(r, dict) or r.get('campaign_root') != str(self.root)]
                other_evidence = [e for e in current['evidence'] if e not in evidence]
                return dict(status=status, classification=latest['classification'], reason=latest['reason'],
                    projects=[*other, *(dict(row, campaign_root=str(self.root)) for row in rows)],
                    evidence=[*other_evidence, *evidence])
            self.store.update_investigation(key, merge=merge)


def retry_spec(selection, target):
    row = selection['selected'][0]
    out = row['outcome']
    if target != 'a3' or 'submission' not in out['task']:
        raise ValueError('feedback retry currently requires an A3 executable optimizer task')
    baseline = out['evidence'].get('baseline')
    if baseline is None:
        raise ValueError('original frozen suite evidence is missing; cannot retry faithfully')
    data = JobStore(baseline['job_root']).read_artifact(baseline['job_id'], baseline['sha256'])
    frozen = json.loads(data)
    manifest = frozen.get('manifest', {})
    if not frozen.get('suite_sha256') or not manifest.get('policy') or not manifest.get('submission'):
        raise ValueError('original frozen suite or comparison policy is missing')
    return dict(feedback_id=row['feedback_id'], task=out['task'],
        suite_sha256=frozen['suite_sha256'], policy=manifest['policy'],
        frozen_submission=manifest.get('submission'),
        max_cases=out['runtime'].get('max_cases') or 24)


def link_retry(job):
    """Link only this fresh campaign; resolution is independently checked by store."""
    if not job.get('retry_feedback_id'):
        return
    store = FeedbackStore(job['feedback_root'])
    store.import_campaign(job['root'])
    original = next(r for r in store.records() if r['feedback_id'] == job['retry_feedback_id'])
    retries = [r for r in store.records() if r['outcome']['source_root'] == str(Path(job['root']).resolve())
               and r['outcome']['task'] == original['outcome']['task']]
    def merge(current):
        ids = list(dict.fromkeys([*current['retries'], *(r['feedback_id'] for r in retries)]))
        updates = dict(retries=ids)
        if current['status'] != 'resolved' and any(r['outcome']['verified'] and (
                original['outcome']['category'] != 'verified-slow' or
                r['outcome']['category'] == 'verified-success') for r in retries):
            updates['status'] = 'resolved'
        return updates
    store.update_investigation(original['feedback_id'], merge=merge)
