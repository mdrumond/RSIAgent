"""Read-only wave lifecycle projection, including siblings not yet dispatched."""
import json
from pathlib import Path

from rsi.jobs import _alive
from rsi.job_inspection import inspect_workflow


def inspect_waves(cycle):
    cycle = Path(cycle)
    path = cycle / 'cycle.json'
    if not path.exists():
        return []
    scheduler = json.loads(path.read_text())
    waves = [*scheduler.get('wave_history', []), scheduler.get('active_wave')]
    output = []
    for wave in waves:
        if not wave:
            continue
        root = cycle / 'waves' / wave['id']
        path = root / 'wave.json'
        state = json.loads(path.read_text()) if path.exists() else {'projects': {}, 'stopped': False}
        projects = []
        for op in wave['operations']:
            row = state['projects'].get(op['id'], {})
            branch = root / 'projects' / op['id']
            path = branch / 'worker.json'
            worker = json.loads(path.read_text()) if path.exists() else {}
            alive = bool(worker.get('process') and _alive(worker['process']))
            status = row.get('status', 'reserved')
            if status == 'pending':
                status = 'running' if alive else 'observation_required'
            result = row.get('receipt', {}).get('result', {})
            outcome = result.get('status')
            admission_path = branch / 'worker-admission.json'
            admission = json.loads(admission_path.read_text()).get('receipt') if admission_path.exists() else None
            if not outcome and admission and admission.get('admitted') is False:
                outcome = ('admission_pending' if admission.get('failure', {}).get('type') in
                           ('probe_pending', 'probe_unknown', 'submission_unknown') else 'admission_denied')
            if row.get('receipt', {}).get('terminal_failure_proof') and outcome == 'suspended':
                outcome = 'failed'
            path = branch / 'runner' / 'workflow.json'
            workflow = json.loads(path.read_text()) if path.exists() else {}
            pending = workflow.get('pending')
            # Wave allowance is the scheduler ceiling, not the runner's admitted
            # allocation. Recovery grants and elapsed time belong to Workflow.
            accounting = inspect_workflow(workflow) if workflow else {}
            project = dict(operation_id=op['id'], project_id=op['original_id'],
                status=status, outcome=outcome, device=row.get('device'),
                assigned_device=row.get('lease_device'), dispatch_id=row.get('dispatch_id'),
                stage=workflow.get('stage'), pending=pending, admission=admission,
                actor_seconds=accounting.get('actor_union_seconds'),
                accounting_state=accounting.get('accounting_state', 'not_started'),
                budget_source='workflow' if workflow else 'unavailable',
                recovery_authorized=row.get('recovery') is not None)
            if 'actor_allocation' in accounting:
                project['actor_allocation'] = accounting['actor_allocation']
            if 'actor_seconds' in accounting.get('remaining', {}):
                project['remaining_actor_seconds'] = accounting['remaining']['actor_seconds']
            if 'allowance' in row:
                project['wave_actor_ceiling'] = row['allowance']
            projects.append(project)
        stopped = state.get('stopped') or (root / 'stop.json').exists()
        status = ('observation_required' if any(p['status'] == 'observation_required' for p in projects)
                  else 'draining' if stopped and any(p['status'] == 'running' for p in projects)
                  else 'stopped' if stopped else wave['phase'])
        output.append(dict(wave_id=wave['id'], status=status, projects=projects,
            memory_ref=wave['memory'], skills_ref=wave['skills'],
            publication=state.get('publication', {}).get('identity') if state.get('publication') else None))
    return output
