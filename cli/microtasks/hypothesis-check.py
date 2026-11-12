"""Independent frozen checks for CPU ownership, admission and dtype incidents."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unchanged():
    for name, expected in CONFIG['protectedHashes'].items():
        assert fingerprint(root / name) == expected, ('protected artifact changed', name)

# Held-out construction fixtures are trusted inputs, not model-authored solutions.
if CONFIG['hypothesis'] == 'construction-admissibility':
    for fixture in CONFIG['variants'][1:]:
        for name, text in fixture['files'].items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

unchanged()
spec = importlib.util.spec_from_file_location('candidate', root / 'kernel.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def ownership(report):
    if report['runtime']['expected'] != report['runtime']['observed'] or report['lease'] != 'clear':
        return dict(classification='operator-blocked', action='report-operator')
    if report['trace']:
        origin = report['trace'][-1]['path']
        if origin.startswith(report['candidatePrefix']):
            return dict(classification='candidate-defect', action='repair-source')
        if origin.startswith(report['protectedPrefix']):
            return dict(classification='operator-blocked', action='report-operator')
    return dict(classification='unproven', action='collect-evidence')

def dtype(report):
    if report['required'] not in report['supported']:
        return dict(classification='specification-defect', action='report-operator')
    proof = report['verification']
    if proof and proof['sourceHash'] == report['sourceHash']:
        return dict(classification='verified', action='accept') if proof['passed'] else dict(classification='candidate-defect', action='repair-source')
    if report['error'] and report['requested'] not in report['supported']:
        return dict(classification='candidate-defect', action='repair-source')
    return dict(classification='unproven', action='run-verifier')

for variant in CONFIG['variants']:
    before = copy.deepcopy(variant)
    if CONFIG['hypothesis'] == 'construction-admissibility':
        proposal = module.repair_proposal(copy.deepcopy(variant['proposal']))
        assert isinstance(proposal, dict)
        for key in ('goal', 'performance_goal'):
            assert proposal.get(key) == variant['proposal'].get(key), ('changed workload', key)
        for key in ('command', 'performance_command', 'baseline_command'):
            assert proposal.get('verifier', {}).get(key) == variant['proposal']['verifier'][key], ('changed target contract', key)
        for name, text in variant['files'].items():
            assert (root / name).read_text() == text, ('changed fixture', name)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=root) as request:
            json.dump(proposal, request)
            request.flush()
            argv = [CONFIG['node'], CONFIG['oracle'], '--root', str(root), '--proposal', request.name]
            if CONFIG['kind'] == 'implemented-mode':
                argv += ['--control', 'baseline']
            result = subprocess.run(argv, capture_output=True, text=True, timeout=12)
            assert result.returncode == 0, result.stderr
            verdict = json.loads(result.stdout.strip().splitlines()[-1])
            assert verdict['admitted'], verdict
    else:
        operation = module.route if CONFIG['hypothesis'] == 'fault-ownership' else module.triage
        observed = operation(variant)
        expected = ownership(before) if CONFIG['hypothesis'] == 'fault-ownership' else dtype(before)
        assert observed == expected, (before, observed, expected)
        assert variant == before, 'report mutated'
    unchanged()
print(json.dumps({'correct': True, 'case': CONFIG['id'], 'variants': len(CONFIG['variants'])}))
