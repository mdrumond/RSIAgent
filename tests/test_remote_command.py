import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from benchmarks.ascend.remote_command import launcher_source, payload_digest, validate_receipt


def payload(tmp_path, command, *, seconds=3, boot=None, wall_shift=0):
    value = dict(argv=list(command), result_directory=str(tmp_path/'result'),
        expires_at=time.time()+seconds+wall_shift, remote_deadline=time.monotonic()+seconds,
        remote_boot_id=boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        binding=dict(route=dict(physical_device=11, target='fixture', runtime='named'),
                     workspace=str(tmp_path), argv=list(command), operation_nonce='unique-command'))
    value['payload_sha256'] = payload_digest(value)
    return value


def run(tmp_path, value, **options):
    script = tmp_path/'command.sh'
    script.write_text(launcher_source(value))
    return subprocess.run(('bash', str(script)), capture_output=True, text=True, timeout=17, **options)


def terminal(value):
    root = Path(value['result_directory'])
    return dict(available=True, receipt=json.loads((root/'terminal.json').read_text()),
        **{name:base64.b64encode((root/(name+'.bin')).read_bytes()).decode() for name in ('stdout','stderr')})


@pytest.mark.parametrize('wall_shift', [-3600,3600])
def test_supervisor_uses_remote_elapsed_domain_and_retains_actual_exit_and_masks(tmp_path, wall_shift):
    command = (sys.executable,'-c',"import os,sys; print(os.environ['ASCEND_RT_VISIBLE_DEVICES']); print(os.environ['ASCEND_VISIBLE_DEVICES'],file=sys.stderr); sys.exit(7)")
    value = payload(tmp_path, command, wall_shift=wall_shift)
    result = run(tmp_path, value)
    assert result.returncode == 7
    assert validate_receipt(terminal(value), value) == dict(returncode=7,stdout='11\n',stderr='11\n')


@pytest.mark.parametrize('boot_changed', [False,True])
def test_late_or_rebooted_dispatch_never_starts_payload(tmp_path, boot_changed):
    marker = tmp_path/'started'
    value = payload(tmp_path, (sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).touch()'),
        seconds=-1 if not boot_changed else 3, boot='different-boot' if boot_changed else None)
    assert run(tmp_path,value).returncode == 124
    assert not marker.exists()
    proof = terminal(value)
    assert proof['receipt']['termination'] == ('boot-changed' if boot_changed else 'deadline')
    assert validate_receipt(proof,value)['returncode'] == 124


def test_broken_stdout_keeps_complete_durable_output(tmp_path):
    value = payload(tmp_path,(sys.executable,'-c',"import sys;sys.stdout.write('x'*200000);sys.stderr.write('exact error')"))
    script = tmp_path/'command.sh';script.write_text(launcher_source(value))
    process = subprocess.Popen(('bash',str(script)),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    process.stdout.close()
    assert process.wait(timeout=5) == 125
    process.stderr.close()
    recovered = validate_receipt(terminal(value),value)
    assert recovered == dict(returncode=0,stdout='x'*200000,stderr='exact error')


def test_hup_reaps_payload_and_descendants_before_terminal_receipt(tmp_path):
    ready=tmp_path/'ready'
    descendant="import pathlib,os,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
    command=(sys.executable,'-c',f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{descendant!r},{str(ready)!r}]);time.sleep(60)")
    value=payload(tmp_path,command,seconds=60)
    script=tmp_path/'command.sh';script.write_text(launcher_source(value))
    process=subprocess.Popen(('bash',str(script)),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        end=time.monotonic()+5
        while not ready.exists() and time.monotonic()<end:time.sleep(.01)
        assert ready.exists()
        pid=int(ready.read_text())
        process.send_signal(signal.SIGHUP)
        assert process.wait(timeout=5)==124
        assert not Path(f'/proc/{pid}').exists()
        proof=terminal(value)
        assert proof['receipt']['process_group_reaped'] is True
        assert validate_receipt(proof,value)['returncode']==124
    finally:
        if process.poll() is None:process.kill();process.wait()
        process.stdout.close();process.stderr.close()


@pytest.mark.parametrize('change',['nonce','argv','output','settlement'])
def test_terminal_proof_rejects_wrong_identity_or_incomplete_stream(tmp_path,change):
    value=payload(tmp_path,(sys.executable,'-c',"print('complete')"))
    assert run(tmp_path,value).returncode==0
    proof=terminal(value)
    if change=='nonce':proof['receipt']['binding']['operation_nonce']='other-command'
    if change=='argv':proof['receipt']['binding']['argv']=['another-command']
    if change=='output':proof['stdout']=base64.b64encode(b'partial').decode()
    if change=='settlement':proof['receipt']['process_group_reaped']=False
    with pytest.raises(ValueError):validate_receipt(proof,value)


def test_duplicate_launcher_does_not_execute_payload_twice(tmp_path):
    count=tmp_path/'count'
    command=(sys.executable,'-c',f"from pathlib import Path;p=Path({str(count)!r});p.write_text(p.read_text()+'x' if p.exists() else 'x')")
    value=payload(tmp_path,command)
    assert run(tmp_path,value).returncode==0
    before=terminal(value)
    assert run(tmp_path,value).returncode==125
    assert count.read_text()=='x'
    assert terminal(value)==before
