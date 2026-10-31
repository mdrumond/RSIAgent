"""Host-only diagnostic templates, immutable parents and real canonical CPU work."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rsi.contracts import BudgetLimits, ProjectSpec
from rsi.curriculum_materialization import (
    MaterializationRejected, capability_catalog, materialize_child, read_materialized_child)
from rsi.project_runner import ProjectConfig, ProjectRunner
from rsi.python_execution import atomic_json
from rsi.skills import builtin_bundle
from test_rsi_project_runner import Session


@pytest.fixture
def cpu(tmp_path):
    source = tmp_path/'parent-source'
    (source/'reference-skills').mkdir(parents=True)
    (source/'reference-skills/guide.txt').write_text('The independent host oracle owns expected values.\n')
    (source/'solution.py').write_text('def solve(xs): return sum(xs)\n')
    cases = source/'cases.json'
    atomic_json(cases, dict(project_id='original', cases=[
        dict(input=[1, 2], expected=3), dict(input=[4, -2], expected=2), dict(input=[], expected=0)]))
    verifier = tmp_path/'parent-oracle.py'
    verifier.write_text("import argparse,json,runpy\np=argparse.ArgumentParser();p.add_argument('--cases')\n"
        "a=p.parse_args();solve=runpy.run_path('solution.py')['solve']\n"
        "for case in json.load(open(a.cases))['cases']: assert solve(case['input'])==case['expected']\n")
    command = [sys.executable, '{verifier}', '--cases', 'cases.json']
    config = ProjectConfig(str(source), str(verifier), dict(
        compile=[sys.executable, '-m', 'py_compile', 'solution.py'], test=command, validation=command))
    spec = ProjectSpec('original', 'Solve original full cases', tuple(command),
                       budgets=BudgetLimits(actor_seconds=60, wall_seconds=90),
                       starting_folder=str(source), correctness_ref=str(cases))
    idea = SimpleNamespace(name='single shape', goal='Investigate one simpler case',
        acceptance='Same independent oracle passes', template_id='cpu.case-subset',
        parameters={'case_indices': [1]}, parent_project_id='original')
    return spec, config, idea


def build(cpu, root, **overrides):
    spec, config, idea = cpu
    return materialize_child(spec, config, idea, root=root, remaining=overrides.pop('remaining', 17.),
        evidence_refs=overrides.pop('evidence_refs', ('a'*64,)), operation_id=overrides.pop('operation_id', 'research-1'),
        **overrides)


def test_cpu_child_runs_all_canonical_stages_with_smaller_frozen_cases(cpu, tmp_path):
    spec, config, idea = cpu
    original = {p: p.read_bytes() for p in Path(config.source).rglob('*') if p.is_file()}
    child = build(cpu, tmp_path/'child')
    assert child.spec.project_id != spec.project_id
    assert child.spec.command == tuple(child.config.commands['test'])
    assert child.spec.budgets.actor_seconds == 17
    assert child.provenance['evidence_refs'] == ['a'*64]
    assert child.provenance['setup_excluded_from_actor_time'] is True
    assert json.loads(Path(child.spec.correctness_ref).read_text())['cases'] == [dict(input=[4, -2], expected=2)]
    assert Path(child.config.verifier).read_bytes() == Path(config.verifier).read_bytes()
    assert (Path(child.config.source)/'reference-skills/guide.txt').read_bytes() == original[Path(config.source)/'reference-skills/guide.txt']
    runner = ProjectRunner.create(tmp_path/'runner', child.spec, builtin_bundle(), child.config,
        query=lambda _: dict(results=[], negative_records=[], provenance={'test': True}),
        backend_identity='explicit-child-test', test_backend=True, session_factory=lambda: Session(repair=False))
    result = runner.run()
    assert result['status'] == 'accepted'
    assert len(result['history']) == 9
    assert all(p.read_bytes() == data for p, data in original.items())
    assert len(json.loads(Path(spec.correctness_ref).read_text())['cases']) == 3


def test_receipt_idempotence_and_operation_identity(cpu, tmp_path):
    root = tmp_path/'child'
    child = build(cpu, root)
    again = build(cpu, root, remaining=2)
    assert again == child  # Existing admission is not another allocation.
    receipt = child.provenance['receipt']
    assert read_materialized_child(receipt['path'], sha256=receipt['sha256']) == child
    assert build(cpu, tmp_path/'another', operation_id='research-2').spec.project_id != child.spec.project_id
    with pytest.raises(MaterializationRejected):
        build(cpu, root, operation_id='reused-root')
    Path(child.spec.correctness_ref).write_text('{}')
    with pytest.raises(MaterializationRejected):
        build(cpu, root)


@pytest.mark.parametrize('indices', [[], [0, 1, 2], [True], [-1], [3], [1, 1]])
def test_invalid_or_not_simpler_case_choice_has_no_materialization(cpu, tmp_path, indices):
    cpu[2].parameters = dict(case_indices=indices)
    root = tmp_path/'rejected'
    with pytest.raises(MaterializationRejected):
        build(cpu, root)
    assert not root.exists()


def test_explicit_remaining_not_parent_default_and_partial_setup_never_overwritten(cpu, tmp_path):
    spec, config, idea = cpu
    limits = iter([12., 7.])
    child = build(cpu, tmp_path/'child', remaining=lambda: next(limits))
    assert child.spec.budgets.actor_seconds == child.spec.budgets.wall_seconds == 7
    assert spec.budgets.actor_seconds == 60
    for remaining in (0, -1, float('inf'), None):
        with pytest.raises(MaterializationRejected):
            build(cpu, tmp_path/'no-time', remaining=remaining)
    root = tmp_path/'partial'
    root.mkdir()
    (root/'partial-evidence').write_text('must remain')
    with pytest.raises(MaterializationRejected):
        build(cpu, root)
    assert (root/'partial-evidence').read_text() == 'must remain'


def test_catalog_and_unknown_template_or_arbitrary_commands_are_rejected(cpu, tmp_path):
    spec, config, idea = cpu
    catalog = capability_catalog(spec, config)
    assert catalog[0]['parameters']['case_indices']['items']['enum'] == [0, 1, 2]
    idea.parameters['command'] = ['sh', '-c', 'anything']
    with pytest.raises(MaterializationRejected):
        build(cpu, tmp_path/'bad-parameters')
    idea.parameters = {'case_indices': [0]}
    idea.template_id = 'unsupported'
    with pytest.raises(MaterializationRejected):
        build(cpu, tmp_path/'unsupported')
    idea.template_id = 'cpu.case-subset'
    idea.parent_project_id = 'different-original'
    with pytest.raises(MaterializationRejected):
        build(cpu, tmp_path/'wrong-parent')
    assert not list(tmp_path.glob('bad-*'))


def _native_parent(tmp_path, case_rows=None):
    from rsi.triton_worker_binding import prepare_worker_bindings
    source = tmp_path/'native-parent'
    source.mkdir()
    (source/'candidate.py').write_text('def candidate(*args): return args[0]\n')
    (source/'full-resource.txt').write_text('Preserve all original source context\n')
    cases = tmp_path/'native-cases.json'
    atomic_json(cases, dict(project_id='native-original', operation='add', cases=[
        dict(case_id=str(i), shape=[i+1], seed=i, dtype='float32', layout='contiguous')
        for i in range(3)] if case_rows is None else case_rows))
    devices = {}
    for device in (0, 1):
        runtime = tmp_path/f'runtime-{device}.json'
        atomic_json(runtime, dict(physical_device=device, device_health=dict(product='a3', chip=0)))
        identity = f'profile-{device}'
        devices[str(device)] = dict(runtime_manifest=str(runtime),executor_identity=identity,python='python3',
            metadata=dict(target='fixture',runtime='named',physical_device=device,executor_identity=identity))
    parent = ProjectSpec('native-original','Preserve full original workload',('template',),
        budgets=BudgetLimits(actor_seconds=60),starting_folder=str(source),correctness_ref=str(cases))
    old_binding = prepare_worker_bindings([dict(spec=parent,source=str(source),cases=str(cases))],devices,tmp_path/'old.json')
    old_bytes = Path(old_binding['manifest']).read_bytes()
    variant = json.loads(old_bytes)['projects'][parent.project_id]['variants']['0']
    parent, config = ProjectSpec.from_dict(variant['spec']), ProjectConfig.from_dict(variant['config'])
    return parent, config, old_binding, devices, cases


def test_new_triton_id_has_own_device_table_and_resolves_actual_lease(tmp_path):
    from rsi.project_wave import WaveRequest
    from rsi.triton_worker_binding import bind_project
    parent, config, old_binding, devices, cases = _native_parent(tmp_path)
    old_bytes = Path(old_binding['manifest']).read_bytes()
    idea = dict(name='tail',goal='Investigate one tail',acceptance='Independent oracle',
        template_id='triton.case-subset',parameters=dict(case_indices=[1]))
    child = materialize_child(parent,dict(config=asdict(config),binding=old_binding),idea,
        root=tmp_path/'native-child',remaining=9,evidence_refs=('b'*64,),operation_id='diagnose')
    binding = child.provenance['binding']
    table = json.loads(Path(binding['manifest']).read_text())
    assert set(table['projects']) == {child.spec.project_id}
    assert set(table['projects'][child.spec.project_id]['variants']) == {'0','1'}
    branch = tmp_path/'leased-worker'
    branch.mkdir()
    request = WaveRequest(root=branch/'runner',project_id='canonical-child-operation',device='1',
        dispatch_id='dispatch',reopen=False,memory=b'',skills=b'',actor_seconds=9.,actor_elapsed=0.)
    atomic_json(branch/'worker.json',dict(pid=os.getpid(),dispatch_id='dispatch'))
    admission = dict(operation_id=request.project_id,spec=child.spec.to_dict(),config=asdict(child.config))
    resolution = bind_project(request,admission,metadata=devices['1']['metadata'],configuration=binding,remaining=lambda:9.)
    assert resolution['spec']['project_id'] == child.spec.project_id
    assert resolution['spec']['budgets']['actor_seconds'] == 9
    assert resolution['evidence']['physical_device'] == '1'
    assert Path(old_binding['manifest']).read_bytes() == old_bytes
    assert len(json.loads(cases.read_text())['cases']) == 3
    assert (Path(resolution['config']['source'])/'full-resource.txt').read_text() == 'Preserve all original source context\n'


def test_single_case_shape_forks_preserve_template_oracle_and_can_shrink_again(tmp_path):
    case = dict(case_id='matrix',shape=[257,33],seed=7,dtype='float32',layout='transpose',atol=1e-5,rtol=1e-4)
    parent, config, binding, _, cases = _native_parent(tmp_path, [case])
    original = cases.read_bytes()
    for number, shape in enumerate(([2,4],[1,1])):
        wrapper = dict(config=asdict(config),binding=binding)
        catalog = capability_catalog(parent,wrapper)
        template = next(row for row in catalog if row['template_id']=='triton.shape-reduction.case-0')
        assert template['parameters']['shape']['minItems'] == template['parameters']['shape']['maxItems'] == 2
        assert len(template['parameters']['shape']['items']['enum']) <= 12
        idea = dict(name='tiny',goal='Investigate a tiny input',acceptance='Existing independent oracle',
            template_id=template['template_id'],parameters=dict(shape=shape))
        child = materialize_child(parent,wrapper,idea,root=tmp_path/f'shape-{number}',
            remaining=8.,evidence_refs=('c'*64,),operation_id=f'shape-{number}')
        contract = json.loads(Path(child.spec.correctness_ref).read_text())
        assert contract['project_id']=='native-original' and contract['operation']=='add'
        assert contract['cases']==[dict(case,shape=shape)]
        assert Path(child.config.verifier).read_bytes()==Path(config.verifier).read_bytes()
        assert child.spec.project_id!=parent.project_id
        parent, config, binding = child.spec,child.config,child.provenance['binding']
    assert capability_catalog(parent,dict(config=asdict(config),binding=binding)) == []
    assert cases.read_bytes()==original


@pytest.mark.parametrize('shape', [[4,8],[1],[1,1,1],[0,1],[-1,1],[True,1],[8,1],[1,16],[3,1]])
def test_shape_choice_requires_rank_positive_advertised_per_axis_strict_reduction(tmp_path,shape):
    parent,config,binding,_,cases=_native_parent(tmp_path,[dict(shape=[4,8],seed=0,dtype='float32',layout='contiguous')])
    idea=dict(name='tiny',goal='A smaller input',acceptance='Same oracle',
        template_id='triton.shape-reduction.case-0',parameters=dict(shape=shape))
    root=tmp_path/'rejected-shape'
    with pytest.raises(MaterializationRejected):
        materialize_child(parent,dict(config=asdict(config),binding=binding),idea,
            root=root,remaining=9.,evidence_refs=('d'*64,),operation_id='bad-shape')
    assert not root.exists() and json.loads(cases.read_text())['cases'][0]['shape']==[4,8]
