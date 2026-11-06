import copy
import json
import pytest
from ascend_tools.baselines import WORKLOADS, describe, validate_receipt


def test_frozen_contracts_cover_all_public_cases_and_reject_tampering(tmp_path, monkeypatch):
    import ascend_tools.baselines as baselines
    assert {key:len(describe(key)['caseIds']) for key in WORKLOADS} == dict(
        fa=20,gdn=50,bsa=50,kimi_delta_attention=20,lact_ttt_chunk=20,
        sparse_lightning_indexer_grad_kl_loss_enhance=20,tensor_program_vm=20)
    import shutil
    shutil.copytree(baselines.ROOT,tmp_path/'packs')
    monkeypatch.setattr(baselines,'ROOT',tmp_path/'packs')
    p=tmp_path/'packs/upstream/tasks/level5/kimi_delta_attention/golden.py'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Frozen reference changed'):
        describe('kimi_delta_attention')


def test_receipt_rejects_partial_wrong_runtime_and_invalid_timing():
    expected={**describe('fa'),'target':'a3','runtimeFingerprint':'runtime'}
    receipt={k:expected[k] for k in ['workload','target','sourceHash','caseHash','verifierHash']}
    receipt.update(format=1,qualified=True,runtime={'fingerprint':'runtime'},
        independentChecks={'passed':True,'checkerHash':expected['independentCheckHash']},
        cases=[dict(id=i,passed=True,referenceHash='a',outputHash='b',maxAbsError=0.) for i in expected['caseIds']],
        timing={'passed':True,'samples':[{'id':i,'seconds':[.001]} for i in expected['caseIds']]})
    assert validate_receipt(receipt,expected) is receipt
    for mutate in [lambda d:d['cases'].pop(),lambda d:d['runtime'].update(fingerprint='other'),
                   lambda d:d['timing']['samples'][0].update(seconds=[float('nan')]),
                   lambda d:d.update(target='a5')]:
        bad=copy.deepcopy(receipt);mutate(bad)
        with pytest.raises(ValueError):validate_receipt(bad,expected)


def test_contract_identity_binds_tolerances(tmp_path,monkeypatch):
    import ascend_tools.baselines as b, shutil
    before=describe('fa');shutil.copytree(b.ROOT,tmp_path/'packs');monkeypatch.setattr(b,'ROOT',tmp_path/'packs')
    p=b.ROOT/'manifest.json';m=json.loads(p.read_text());m['fa']['thresholds']['float32']=.5;p.write_text(json.dumps(m))
    after=describe('fa')
    assert after['sourceHash']!=before['sourceHash'] and after['caseHash']==before['caseHash']
