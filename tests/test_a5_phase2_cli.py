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


def test_cli_rejects_snapshot_path_or_cell_identity_before_execution(tmp_path):
    composition, _source, destination, snapshot = _snapshot(tmp_path)
    dependencies = _dependencies(snapshot)
    source_called = False

    def no_source(_request):
        nonlocal source_called
        source_called = True
        raise AssertionError("identity mismatch reached candidate execution")

    dependencies = Phase2Dependencies(destination, no_source, dependencies.verify)
    root = tmp_path / "phase2"

    with pytest.raises(ValueError, match="snapshot path"):
        main(
            [
                "run",
                "--root",
                str(root),
                "--snapshot",
                str(tmp_path / "wrong-snapshot"),
                "--cell-id",
                destination.cell_id,
            ],
            runtime_factory=lambda _args: (snapshot, dependencies),
        )

    alternative = next(
        cell
        for cell in phase2_cells(
            destination.guide, composition.qualification.report_sha256
        )
        if cell != destination
    )
    with pytest.raises(ValueError, match="requested cell"):
        main(
            [
                "run",
                "--root",
                str(root),
                "--snapshot",
                str(snapshot.root),
                "--cell-id",
                alternative.cell_id,
            ],
            runtime_factory=lambda _args: (snapshot, dependencies),
        )

    wrong_dependencies = Phase2Dependencies(
        alternative, no_source, dependencies.verify
    )
    with pytest.raises(ValueError, match="runtime cell"):
        main(
            [
                "run",
                "--root",
                str(root),
                "--snapshot",
                str(snapshot.root),
                "--cell-id",
                destination.cell_id,
            ],
            runtime_factory=lambda _args: (snapshot, wrong_dependencies),
        )

    assert source_called is False
    assert not (root / "cells" / destination.cell_id / "protocol.json").exists()


def test_cli_fake_runtime_runs_one_cell_end_to_end(tmp_path, capsys):
    _composition, _source, destination, snapshot = _snapshot(tmp_path)
    dependencies = _dependencies(snapshot)
    factory_calls = []

    def factory(args):
        factory_calls.append(args)
        return snapshot, dependencies

    root = tmp_path / "phase2"
    assert (
        main(
            [
                "run",
                "--root",
                str(root),
                "--snapshot",
                str(snapshot.root),
                "--cell-id",
                destination.cell_id,
            ],
            runtime_factory=factory,
        )
        == 0
    )

    output = capsys.readouterr().out
    terminal = json.loads(output)
    terminal_path = root / "cells" / destination.cell_id / "terminal.json"
    assert output == json.dumps(
        terminal, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert terminal["status"] == "passed"
    assert terminal["cell_id"] == destination.cell_id
    stored = json.loads(terminal_path.read_text(encoding="utf-8"))
    explanations = terminal.pop("status_explanations")
    assert stored == terminal
    assert "status_explanations" not in stored
    assert explanations[0]["code"] == stored["status"] == "passed"
    assert "configured acceptance checks" in explanations[0]["description"]
    # Display additions must not invalidate the original terminal record identity.
    from benchmarks.a5kernels.evidence import canonical_digest
    assert stored["record_sha256"] == canonical_digest({
        key: value for key, value in stored.items() if key != "record_sha256"})
    assert len(factory_calls) == 1
    assert factory_calls[0].cell_id == destination.cell_id
    assert factory_calls[0].snapshot == snapshot.root


def test_cli_default_factory_keeps_production_composition_explicit(tmp_path):
    with pytest.raises(RuntimeError, match="production dependencies"):
        main(
            [
                "run",
                "--root",
                str(tmp_path / "phase2"),
                "--snapshot",
                str(tmp_path / "snapshot"),
                "--cell-id",
                "a5-phase2-cell-uncomposed",
            ]
        )
