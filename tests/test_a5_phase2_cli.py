from __future__ import annotations

import json

import pytest

from benchmarks.a5kernels.phase2_attention import (
    AttentionReceipt,
    AttentionTargetVerifier,
    attention_reference,
)
from benchmarks.a5kernels.phase2_live import Phase2Dependencies
from benchmarks.a5kernels.phase2_memory import phase2_cells
from run_a5_phase2 import main
from tests.test_a5_phase2_memory import _snapshot


SOURCE = """\
import catlass.tla as tla

@tla.kernel
def dense_attention(q, k, v, output):
    pass
"""


class _PassingBackend:
    def execute(self, source, execution, workdir):
        del source, workdir
        return AttentionReceipt(
            0,
            attention_reference(execution),
            session_handle="bz-a5:cli-test",
        )


def _dependencies(snapshot):
    verifier = AttentionTargetVerifier(_PassingBackend())
    return Phase2Dependencies(
        snapshot.destination,
        lambda request: SOURCE + f"\n# attempt {request.attempt}\n",
        lambda candidate, request_id, attempt_id, workdir: verifier.verify(
            candidate,
            request_id=request_id,
            attempt_id=attempt_id,
            execution_profile="bz-a5",
            workdir=workdir,
        ),
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--root", "out", "--snapshot", "snapshot"],
        [
            "run",
            "--root",
            "out",
            "--snapshot",
            "snapshot",
            "--cell-id",
            "first",
            "--cell-id",
            "second",
        ],
        [
            "run",
            "--root",
            "out",
            "--snapshot",
            "snapshot",
            "--cell-id",
            "cell",
            "--unknown",
        ],
    ],
)
def test_cli_rejects_missing_duplicate_and_unknown_cell_forms(argv):
    factory_called = False

    def factory(_args):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("invalid arguments reached the runtime factory")

    with pytest.raises(SystemExit) as error:
        main(argv, runtime_factory=factory)

    assert error.value.code == 2
    assert factory_called is False


@pytest.mark.parametrize('command', ['run', 'qualify'])
def test_retired_phase2_launch_never_composes_or_executes(tmp_path, command):
    calls = []
    with pytest.raises(SystemExit) as error:
        main([command, '--root', str(tmp_path / 'out'), '--snapshot', 'absent',
              '--cell-id', 'historical'], runtime_factory=lambda args: calls.append(args))
    assert error.value.code == 2
    assert calls == []
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('target', ['a5', 'a3'])
def test_historical_phase2_report_reads_retained_results(tmp_path, capsys, target):
    terminal = tmp_path / 'cells' / 'old-cell' / 'terminal.json'
    terminal.parent.mkdir(parents=True)
    raw = json.dumps({'cell_id': 'old-cell', 'status': 'passed'}).encode()
    terminal.write_bytes(raw)
    if target == 'a3':
        from benchmarks.ascend.phase2_cli import main as report
        args = ['--target', 'a3']
    else:
        report, args = main, []
    assert report([*args, 'report', '--root', str(tmp_path)]) == 0
    record, = json.loads(capsys.readouterr().out)['records']
    explanations = record.pop('status_explanations')
    assert record == json.loads(raw)
    assert explanations[0]['code'] == record['status']
    assert terminal.read_bytes() == raw
