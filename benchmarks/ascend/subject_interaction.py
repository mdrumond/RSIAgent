"""Retained manual continuations and reviewed restarts for subject campaigns."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from optimizer.jobs import atomic, canonical
from .actor_control import ActorControl, grant_elapsed
from .subject_campaign import _effective_project


def _read(path, default=None):
    return json.loads(Path(path).read_bytes()) if Path(path).exists() else default


@contextmanager
def _lock(path, *, wait=True):
    with Path(path).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        yield


def binding(config_path, campaign_id):
    config = _read(config_path, {})
    if config.get('controls', {}).get('enabled') is not True:
        raise PermissionError('dashboard controls are disabled')
    row = next(row for row in config.get('subject_campaigns', []) if row['id'] == campaign_id)
    if row.get('target') not in {'a3', 'a5'}:
        raise ValueError('subject campaign requires a registered target binding')
    return row, Path(row['root'])


def _project(root, project_id, request_id=None):
    state = _read(root / 'subject-campaign.json')
    active = _read(root / 'memory/active.json', {})
    published = project_id in active.get('admissions', {})
    for version in active.get('active', {}).values():
        memory = _read(root / 'memory/versions' / (version + '.json'), {})
        published = published or memory.get('provenance', {}).get('project_id') == project_id
    if published:
        raise ValueError('project is already published into memory')
    pending = state.get('pending') or {}
    record = state['projects'].get(project_id)
    if record is None:
        item = next(item for item in pending.get('items', []) if item['project_id'] == project_id)
        record = dict(plan=item, objective=pending['subject']['objective'])
    if record.get('superseded_by') and record.get('cancellation', {}).get('request_id') != request_id:
        raise ValueError('project was superseded; continue ' + record['superseded_by'])
    if record.get('directory'):
        wave = None
    elif project_id in pending.get('projects', []):
        wave = pending['wave_id']
    else:
        wave = state['waves'][record['wave_index'] - 1]
    directory = root / record['directory'] if record.get('directory') else root / 'waves' / wave / project_id
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('project directory is outside campaign')
    return state, record, directory


def _inbox(directory, record):
    if record['plan'].get('task', {}).get('kind') == 'optimizer':
        jobs = sorted((directory / 'jobs').glob('*/manifest.json'))
        if jobs:
            return ActorControl(jobs[0].parent)
    return ActorControl(directory / 'actor')


def _seconds(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError('provide a finite positive additional elapsed allowance')


def _launch(config_path, campaign_id, request):
    _, root = binding(config_path, campaign_id)
    log = root / 'dashboard-requests' / (request['id'] + '.log')
    with log.open('ab') as output:
        subprocess.Popen([sys.executable, '-m', __name__, '--dashboard-config',
            str(Path(config_path).resolve()), '--campaign', campaign_id, '--request', request['id']],
            cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)


def _unsettled(directory, project_id, exclude):
    for path in directory.glob('*.json'):
        previous = _read(path)
        replacement = previous.get('proposal', {}).get('after', {}).get('item', {}).get('project_id')
        if (previous['id'] != exclude and project_id in {previous['project_id'], replacement}
                and previous['status'] in {'running', 'observation-required'}):
            return previous['id']


def _request(config_path, campaign_id, payload, key, launcher=None):
    if not isinstance(key, str) or not key.strip():
        raise ValueError('Idempotency-Key is required')
    _seconds(payload['additional_seconds'])
    row, root = binding(config_path, campaign_id)
    for field in ('config', 'initial_memory'):
        if not isinstance(row.get(field), str) or not Path(row[field]).is_file():
            raise ValueError('configure subject campaign ' + field + ' before resuming')
    directory = root / 'dashboard-requests'
    directory.mkdir(exist_ok=True)
    identifier = hashlib.sha256(key.encode()).hexdigest()[:32]
    path = directory / (identifier + '.json')
    fingerprint = hashlib.sha256(canonical(payload)).hexdigest()
    with _lock(directory / '.lock'):
        previous_work = _unsettled(directory, payload['project_id'], identifier)
        if previous_work is not None:
            raise ValueError('observe and settle prior manual request ' + previous_work + ' before starting more work')
        previous = _read(path)
        if previous is not None:
            if previous['fingerprint'] != fingerprint:
                raise ValueError('request key reused with different instructions')
            if previous['status'] != 'failed':
                return previous
            request = dict(previous, status='queued')
            request.pop('error', None)
        else:
            request = dict(payload, id=identifier, status='queued', fingerprint=fingerprint,
                           created=time.time())
        atomic(path, canonical(request))
        atomic(root / 'manual-pause.json', canonical(dict(request=identifier)))
    try:
        (launcher or _launch)(config_path, campaign_id, request)
    except Exception:
        request.update(status='failed', error='manual worker could not be launched')
        with _lock(directory / '.lock'):
            atomic(path, canonical(request))
            if _read(root / 'manual-pause.json', {}).get('request') == identifier:
                (root / 'manual-pause.json').unlink(missing_ok=True)
        raise
    return request


def submit_feedback(config_path, campaign_id, project_id, text, key,
                    additional_seconds=None, *, launcher=None):
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > 16384:
        raise ValueError('feedback must contain 1..16384 UTF-8 bytes')
    _, root = binding(config_path, campaign_id)
    state, record, directory = _project(root, project_id)
    inbox = _inbox(directory, record)
    previous = next((message for message in inbox.status()['messages'] if message['key'] == key), None)
    if previous is not None:
        if previous['text'] != text:
            raise ValueError('feedback key reused with different text')
        identifier = hashlib.sha256(key.encode()).hexdigest()[:32]
        existing = request_status(config_path, campaign_id, identifier)
        if existing is None:
            return dict(message=previous, request=None)
    # A completed outcome needs reopening even while another campaign actor runs.
    live = inbox.live()
    if not live:
        _seconds(additional_seconds)
        request = _request(config_path, campaign_id, dict(kind='continue', project_id=project_id,
            text=text, feedback_key=key, additional_seconds=additional_seconds), key, launcher)
    else:
        request = None
    message = inbox.send(key, text)
    return dict(message=message, request=request)


def request_restart(config_path, campaign_id, proposal, additional_seconds, key, *, launcher=None):
    if proposal.get('status') != 'approved' or proposal.get('kind') != 'restart':
        raise ValueError('restart requires an approved curriculum proposal')
    _, root = binding(config_path, campaign_id)
    if _read(root / 'curriculum-edits' / (proposal['id'] + '.json')) != proposal:
        raise ValueError('restart proposal differs from retained curriculum approval')
    identifier = hashlib.sha256(key.encode()).hexdigest()[:32] if isinstance(key, str) else None
    _, record, directory = _project(root, proposal['project'], identifier)
    value = _request(config_path, campaign_id, dict(kind='restart', project_id=proposal['project'],
        proposal=proposal, additional_seconds=additional_seconds), key, launcher)
    _inbox(directory, record).stop()
    return value


def request_status(config_path, campaign_id, identifier):
    _, root = binding(config_path, campaign_id)
    if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('invalid manual request ID')
    path = root / 'dashboard-requests' / (identifier + '.json')
    value = _read(path)
    if value is not None and value['status'] == 'running':
        try:
            with _lock(path.with_suffix('.lock'), wait=False):
                value.update(status='observation-required', error='Prior worker interrupted; observe retained operations before continuing')
                atomic(path, canonical(value))
        except BlockingIOError:
            pass
    return value


def build_campaign(row, root):
    from .subject_runtime import build_subject_campaign
    from .subjects import Subject
    subjects = _read(root / 'subjects.json')['subjects']
    identity = _read(root / 'subject-runtime.json', {})
    return build_subject_campaign(_read(row['config']), target=row['target'], root=root,
        subjects=[Subject.from_mapping(item) for item in subjects],
        initial_memory=Path(row['initial_memory']).read_bytes(),
        max_seconds=identity.get('max_seconds', row.get('max_seconds', 7200)))


def run_request(config_path, campaign_id, identifier, *, builder=build_campaign):
    row, root = binding(config_path, campaign_id)
    path = root / 'dashboard-requests' / (identifier + '.json')
    with _lock(root / '.campaign.lock'), _lock(path.with_suffix('.lock')):
        request = request_status(config_path, campaign_id, identifier)
        if request['status'] == 'succeeded':
            return request
        if request['status'] in {'running', 'observation-required'}:
            request.update(status='observation-required', error='Prior worker interrupted; observe retained operations before continuing')
            atomic(path, canonical(request))
            return request
        previous_work = _unsettled(path.parent, request['project_id'], identifier)
        if previous_work is not None:
            request.update(status='observation-required', error='Prior manual request ' + previous_work + ' needs settlement')
            atomic(path, canonical(request))
            return request
        request.update(status='running', started=time.time(), worker_pid=os.getpid())
        atomic(path, canonical(request))
        budget, campaign = None, None
        try:
            _, original, directory = _project(root, request['project_id'], identifier)
            from .time_budget import AttemptTimeBudget
            elapsed = root / 'elapsed.json'
            identity = _read(root / 'subject-runtime.json', {})
            original_seconds = identity.get('max_seconds', row.get('max_seconds', 7200))
            if not elapsed.exists():
                atomic(elapsed, canonical(dict(max_seconds=original_seconds, elapsed_seconds=0)))
            if _read(elapsed).get('active_since') is not None:
                # The campaign lease proves its old host interval is no longer
                # active. Recover its charged wall time through the budget API.
                AttemptTimeBudget(elapsed, original_seconds).close()
            grant_elapsed(elapsed, identifier, request['additional_seconds'])
            budget = AttemptTimeBudget(elapsed, original_seconds)
            campaign = builder(row, root)
            campaign.budget = budget
            result = _execute(campaign, request, original, directory, path)
            request.update(status='succeeded', result=result, finished=time.time())
        except Exception as error:
            request.update(status='observation-required' if request.get('phase') == 'executing' else 'failed',
                           error=type(error).__name__, finished=time.time())
        finally:
            if campaign is not None and campaign.memory_gate is not None:
                campaign.memory_gate.close()
            if budget is not None:
                budget.close()
            with _lock(path.parent / '.lock'):
                pause = _read(root / 'manual-pause.json', {})
                if pause.get('request') == identifier:
                    (root / 'manual-pause.json').unlink(missing_ok=True)
            atomic(path, canonical(request))
        return request


def _execute(campaign, request, original, directory, request_path):
    root, pid = campaign.root, request['project_id']
    subject_state = campaign.scheduler.state()
    pending = campaign.state.get('pending') or {}
    ordinal = original.get('wave_index', 0)
    original_wave = original.get('origin_wave_id') or (campaign.state['waves'][ordinal - 1]
        if 0 < ordinal <= len(campaign.state['waves']) else None)
    if request['project_id'] in pending.get('projects', []):
        original_wave = pending['wave_id']
    subject_id = original.get('subject_id') or subject_state['waves'].get(original_wave, {}).get('subject_id')
    if subject_id is None and request['project_id'] in pending.get('projects', []):
        subject_id = pending.get('subject', {}).get('subject_id')
    repairs = list(dict.fromkeys([*original.get('repairs', []),
        *original['plan'].get('repairs', []),
        *(key for key, row in subject_state['repairs'].items()
          if row['status'] == 'open' and row['project_id'] == request['project_id'])]))
    checkpoint = _read(directory / 'subject-execution.json', {})
    snapshot_id = request.get('snapshot_sha256') or checkpoint.get('identity', {}).get('memory_sha256')
    if snapshot_id is None:
        ordinal = original.get('wave_index', 0)
        wave_id = campaign.state['waves'][ordinal - 1] if 0 < ordinal <= len(campaign.state['waves']) else pending.get('wave_id')
        snapshot_id = campaign.scheduler.state()['waves'].get(wave_id, {}).get('memory_snapshot_sha256')
        if snapshot_id is None:
            snapshot_id = pending.get('memory_sha256')
        if snapshot_id is None:
            raise ValueError('original frozen project memory snapshot is unavailable')
    snapshot = campaign.artifacts.read_bytes(snapshot_id)
    request['snapshot_sha256'] = snapshot_id
    atomic(request_path, canonical(request))
    item = original['plan']
    if request['kind'] == 'restart':
        item = request['proposal']['after']['item']
        pid = item['project_id']
        existing = campaign.state['projects'].get(pid)
        if existing and existing.get('manual_request') != request['id']:
            raise ValueError('restart project ID was already used')
        campaign.state['projects'][request['project_id']] = dict(original,
            wave_index=original.get('wave_index', len(campaign.state['waves'])+1),
            superseded_by=pid, cancellation=dict(request_id=request['id'], reason='reviewed-restart'))
        directory = root / 'manual-attempts' / request['id'] / pid
        original = dict(objective=original['objective'], plan=item,
            wave_index=original.get('wave_index', 0), restart_of=request['project_id'],
            subject_id=subject_id, supersedes=[request['project_id'], *original.get('supersedes', [])],
            repairs=list(dict.fromkeys([*repairs, *item.get('repairs', [])])))
        ActorControl(directory / 'actor').send(request['id'], request['proposal']['after']['instruction'])
    else:
        # Retain the old immutable result and review before reopening its cache.
        archive = directory / 'manual-history' / request['id']
        archive.mkdir(parents=True, exist_ok=True)
        if not (archive / 'ready.json').exists():
            if not (archive / 'project.json').exists():
                atomic(archive / 'project.json', canonical(original))
            for name in ('subject-execution.json', 'sft-traces'):
                source, destination = directory / name, archive / name
                if source.exists() and not destination.exists():
                    source.replace(destination)
            for source in (directory / 'evidence.jsonl', directory / 'actor/actor-control.json'):
                destination = archive / source.name
                if source.is_file() and not destination.exists():
                    shutil.copy2(source, destination)
            atomic(archive / 'ready.json', canonical(dict(prepared=True)))
        _inbox(directory, original).stop(False)
        _inbox(directory, original).send(request['feedback_key'], request['text'])
        for job_path in (directory / 'jobs').glob('*/manifest.json'):
            from optimizer.jobs import JobStore
            JobStore(job_path.parent.parent).continue_with_feedback(job_path.parent.name,
                additional_seconds=request['additional_seconds'], key=request['id'])
    for job_path in (root / 'memory-reviews').glob('*/job.json'):
        job = _read(job_path)
        if job.get('provenance', {}).get('project_id') == request['project_id'] and job['status'] != 'accepted':
            job.setdefault('original_status', job['status'])
            job.update(status='superseded', superseded_by=request['id'])
            atomic(job_path, canonical(job))
            campaign.scheduler.supersede_review(job['review_id'], dict(request_id=request['id'],
                project_id=request['project_id'], admission_job=str(job_path.relative_to(root))))
    directory.mkdir(parents=True, exist_ok=True)
    attempt_id = pid+'-manual-'+request['id']
    if item.get('task', {}).get('kind') == 'optimizer':
        from optimizer.jobs import JobStore
        for manifest in (directory / 'jobs').glob('*/manifest.json'):
            events = JobStore(manifest.parent.parent).events(manifest.parent.name)
            completed = {row['payload']['attempt_id'] for row in events if row['kind'] == 'phase2-attempt-result'}
            unfinished = [row['payload']['attempt_id'] for row in events
                          if row['kind'] == 'phase2-attempt-request' and row['payload']['attempt_id'] not in completed]
            if unfinished:
                attempt_id = unfinished[-1]
    atomic(directory / 'manual-execution.json', canonical(dict(attempt_id=attempt_id)))
    existing = campaign.state['projects'].get(pid, {})
    if existing.get('manual_request') != request['id']:
        prior_project = campaign.artifacts.put_json(original)
        campaign.state['projects'][pid] = dict(original, plan=item, directory=str(directory.relative_to(root)),
            manual_request=request['id'], prior_project_sha256=prior_project,
            repairs=list(dict.fromkeys([*repairs, *original.get('repairs', [])])))
        campaign.state['projects'][pid].pop('outcome', None)
        campaign.state['projects'][pid].pop('review', None)
    record = campaign.state['projects'][pid]
    record['subject_id'] = subject_id
    record['origin_wave_id'] = original_wave
    campaign._save()
    if 'outcome' not in record:
        request['phase'] = 'executing'
        atomic(request_path, canonical(request))
        record['outcome'] = campaign.execute(item, snapshot, directory,
            remaining_seconds=campaign._remaining()).as_dict()
    request['phase'] = 'reviewing'
    atomic(request_path, canonical(request))
    campaign._save()
    _finish_linked_siblings(campaign, record, snapshot, request, request_path)
    if not record.get('review'):
        campaign._review(pid, record)
    # A distinct review identity preserves earlier A/B decisions and receipts.
    review_id = pid + '-manual-' + request['id']
    record['manual_review_id'] = review_id
    review = record['review']
    _settle_manual_repairs(campaign, pid, record, review_id)
    if review.get('memory') is not None and review['decision'] == 'accept':
        campaign.scheduler.register_manual_review(review_id, subject_id)
        provenance = dict(project_id=pid,
            subject_id=subject_id, request=record['review_request'],
            review_sha256=record['review_sha256'])
        if campaign.memory_gate is None:
            campaign.memory.accept(review_id, review['memory'], provenance)
            campaign.scheduler.resolve_review(review_id, 'accepted')
        else:
            campaign.memory_gate.submit(review_id, review['memory'], provenance,
                record['review_request'], remaining_seconds=campaign._remaining())
            while campaign.memory_gate.future is not None and campaign._remaining() > 0:
                campaign._settle_reviews()
                time.sleep(.1)
    campaign._save()
    return dict(project_id=pid, review_id=review_id, directory=str(directory.relative_to(root)))


def _settle_manual_repairs(campaign, pid, record, review_id):
    """Mirror publication's project repair decisions, including paired probes."""
    review = record['review']
    if review['decision'] == 'repair':
        campaign.scheduler.add_repair(review_id, record['subject_id'], pid,
            review['reason'], [record['review_sha256']])
        return
    if review['decision'] != 'accept':
        return
    pending = campaign.state.get('pending') or {}
    siblings = {pid: record}
    for key, row in campaign.state['projects'].items():
        ordinal = row.get('wave_index', 0)
        wave = row.get('origin_wave_id') or (campaign.state['waves'][ordinal - 1]
            if 0 < ordinal <= len(campaign.state['waves']) else None)
        if key in pending.get('projects', []):
            wave = pending['wave_id']
        if wave == record['origin_wave_id']:
            key, row = _effective_project(campaign.state['projects'], key)
            siblings[key] = row
    if pending.get('wave_id') == record['origin_wave_id']:
        for item in pending.get('items', []):
            key, row = _effective_project(campaign.state['projects'], item['project_id'])
            siblings.setdefault(key, row or dict(repairs=item.get('repairs', [])))
    obligations = campaign.scheduler.state()['repairs']
    for repair_id in record.get('repairs', []):
        if obligations.get(repair_id, {}).get('status') != 'open':
            continue
        linked = [row for row in siblings.values()
                  if repair_id in row.get('repairs', row.get('plan', {}).get('repairs', []))]
        if linked and all((row.get('review') or {}).get('decision') == 'accept' for row in linked):
            campaign.scheduler.resolve_repair(repair_id, [row['review_sha256'] for row in linked])


def _finish_linked_siblings(campaign, record, snapshot, request, request_path):
    """Keep the ordinary paired-probe barrier before manual project review."""
    pending = campaign.state.get('pending') or {}
    if pending.get('wave_id') != record['origin_wave_id']:
        return
    linked = {}
    for item in pending.get('items', []):
        pid = item['project_id']
        if (pid == record['plan']['project_id'] or pid in record.get('supersedes', [])
                or not set(item.get('repairs', [])) & set(record.get('repairs', []))):
            continue
        row = campaign.state['projects'].setdefault(pid, dict(plan=item,
            objective=pending['subject']['objective'], repairs=item.get('repairs', []),
            wave_index=len(campaign.state['waves'])+1))
        pid, row = _effective_project(campaign.state['projects'], pid)
        if pid == record['plan']['project_id']:
            continue
        linked[pid] = row
        if 'outcome' not in row:
            request['phase'] = 'executing'
            atomic(request_path, canonical(request))
            directory = (campaign.root / row['directory'] if row.get('directory')
                         else campaign.root / 'waves' / pending['wave_id'] / pid)
            row['outcome'] = campaign.execute(row['plan'], snapshot, directory,
                remaining_seconds=campaign._remaining()).as_dict()
            campaign._save()
    request['phase'] = 'reviewing'
    atomic(request_path, canonical(request))
    for pid, row in linked.items():
        campaign._review(pid, row)
        if row['review']['decision'] == 'repair':
            campaign.scheduler.add_repair(pid, record['subject_id'], pid,
                row['review']['reason'], [row['review_sha256']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dashboard-config', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    run_request(args.dashboard_config, args.campaign, args.request)


if __name__ == '__main__':
    main()
