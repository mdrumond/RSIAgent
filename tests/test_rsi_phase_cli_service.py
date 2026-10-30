"""Actual host CLI/owned worker/service parent admission with scripted decisions."""
import json
import os
from pathlib import Path
import subprocess
import sys

from test_rsi_services import instance as bound_instance
from rsi import jobs
from rsi.cli import main
from rsi.configuration import set_configuration, show_configuration


def test_cli_owned_job_seeds_accepted_parent_with_current_curriculum(bound_instance, monkeypatch, capsys):
    services = bound_instance
    root = services.root
    parent = services.create_cycle('research', 'phase1')
    try:
        parent.step()  # Real canonical CPU project, including independent verifier.
        parent.step()  # Real accepted-memory publication.
        expected_memory = parent.scheduler.state['memory']
        expected_skills = parent.scheduler.state['skills']
        expected_kdb = parent.binding['kdb']
        parent_path = parent.root
    finally:
        parent.close()
    before = {str(p.relative_to(parent_path)): p.read_bytes()
              for p in parent_path.rglob('*') if p.is_file()}
    current = show_configuration(root)['payload']
    current['phase2_projects'][0]['goal'] = 'Current application curriculum'
    current['prompts']['actor'] = 'Current actor steering'
    current['tuning'].update(every_projects=None, at_wave_end=False)
    set_configuration(root, current)
    source, tests = Path(__file__).parents[1], Path(__file__).parent
    monkeypatch.setattr(jobs, '_machinery', lambda _: source)
    import rsi.instance_dispatch
    monkeypatch.setattr(rsi.instance_dispatch, 'launch_context',
        lambda instance, **kwargs: (instance, source, sys.executable, dict(os.environ)))
    original_popen = subprocess.Popen
    children = []
    # Only the explicit provider/session test seam is injected in the real child.
    # Job creation, inherited flock, worker.main, service and host execution stay real.
    program = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from test_rsi_services import Bindings, Decisions, Roles
from rsi.services import InstanceServices
import rsi.service as service
instance = InstanceServices(Path(sys.argv[3]), environment={}, test_bindings=Bindings(
    lambda _: dict(results=[], negative_records=[], provenance={'test': True}),
    Decisions, Roles, 'explicit-test-binding'))
service.InstanceServices = lambda _: instance
from rsi.job_worker import main
sys.argv = ['worker', *sys.argv[3:6]]
main()
'''
    def worker(argv, **kwargs):
        child = original_popen([sys.executable, '-I', '-c', program,
            str(source), str(tests), *argv[-3:]], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', worker)
    try:
        assert main(['--instance', str(root), 'phase2', 'start', '--cycle', 'application',
                     '--from-cycle', 'research', '--inheritance', 'parent']) == 0
        started = json.loads(capsys.readouterr().out)
        assert len(children) == 1
        assert children[0].wait(timeout=30) == 0, (root / f"runs/jobs/{started['job_id']}.log").read_text()
        retained = jobs.status(root, started['job_id'])
        assert retained['state'] == 'completed' and retained['result']['status'] == 'complete'
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
    binding = json.loads((root / 'runs/application/service.json').read_text())
    assert binding['input_handoff']['parent_cycle'] == 'research'
    assert binding['spec']['baseline_kdb'] == expected_memory
    assert binding['spec']['baseline_skills'] == expected_skills
    assert binding['kdb'] == expected_kdb
    assert binding['spec']['projects'][0]['goal'] == current['phase2_projects'][0]['goal']
    assert binding['config']['payload']['prompts']['actor'] == current['prompts']['actor']
    assert before == {str(p.relative_to(parent_path)): p.read_bytes()
                      for p in parent_path.rglob('*') if p.is_file()}
