from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.a5kernels import (
    A5KernelRunner,
    BZSessionAdapter,
    Language,
    RunRequest,
    fixture_for,
)
from benchmarks.a5kernels.bz import CommandResult, OUTPUT_MARKER
from benchmarks.a5kernels.ascendc_runtime import host_driver


def test_fixture_stages_complete_runtime_deterministically():
    first = fixture_for(Language.ASCEND_C)
    second = fixture_for(Language.ASCEND_C)

    assert first == second
    assert first.argv == ("python", "host_driver.py", "input.json")
    assert first.max_length == 4096
    assert tuple(item.relative_path for item in first.files) == (
        "build.json",
        "host_driver.py",
        "kernel.cpp",
    )
    descriptor = json.loads(first.files[0].content)
    assert descriptor == {
        "arch": "dav-3510",
        "compiler": "bisheng",
        "max_elements": 4096,
        "output": "kernel.so",
        "source": "kernel.cpp",
    }


def test_concrete_plan_keeps_host_attempt_and_retained_result_lifecycle():
    class Backend:
        def execute(self, plan):  # pragma: no cover - prepare only
            raise AssertionError("not called")

    class Executor:
        def __init__(self):
            self.dispatch = None
            self.inspections = []

        def run(self, invocation):
            self.dispatch = invocation
            return CommandResult(0, "observer output")

        def inspect(self, argv):
            self.inspections.append(argv)
            if argv[-1] == "logs":
                return CommandResult(0, f"{OUTPUT_MARKER}[3.0]\n")
            return CommandResult(0, "exit=0", session_handle="bz-a5:ascend")

    plan = A5KernelRunner(Backend()).prepare(
        RunRequest(Language.ASCEND_C.value, length=1), attempt_id="ascend-1"
    )
    executor = Executor()
    receipt = BZSessionAdapter(
        executor, session_wrapper="profiles/bz-a5/session.sh"
    ).execute(plan)

    assert plan.argv == ("python", "host_driver.py", "input.json")
    assert executor.dispatch.stdin == ""
    assert executor.dispatch.argv[-1].endswith("/input.json")
    payload = json.loads(next(
        source.content for source in executor.dispatch.files
        if source.relative_path == "input.json"
    ))
    assert payload == {"input_a": list(plan.input_a), "input_b": list(plan.input_b)}
    assert plan.attempt_id == "ascend-1"
    assert executor.dispatch.argv[2].endswith("-ascend-1")
    assert [argv[-1] for argv in executor.inspections] == ["logs", "result"]
    assert receipt.output == (3.0,)
    assert receipt.session_handle == "bz-a5:ascend"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"input_a": [1], "input_b": [2], "command": "sh"}, "only input_a"),
        ({"input_a": [1], "input_b": [2, 3]}, "equal lengths"),
        ({"input_a": [True], "input_b": [2]}, "numeric JSON array"),
        ({"input_a": [], "input_b": []}, "non-empty"),
    ],
)
def test_vector_input_rejects_commands_and_invalid_shapes(payload, message):
    with pytest.raises(ValueError, match=message):
        host_driver._read_vectors(StringIO(json.dumps(payload)), max_elements=4096)


def test_compile_uses_an_argv_and_fixed_a5_arch(monkeypatch, tmp_path):
    torch_root = tmp_path / "torch"
    npu_root = tmp_path / "torch_npu"
    torch_root.mkdir()
    npu_root.mkdir()
    fake_torch = SimpleNamespace(
        __file__=str(torch_root / "__init__.py"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    fake_torch_npu = SimpleNamespace(__file__=str(npu_root / "__init__.py"))
    (tmp_path / "kernel.cpp").write_text("// staged", encoding="utf-8")
    calls = []

    monkeypatch.setattr(host_driver.shutil, "which", lambda name: "/opt/cann/bin/bisheng")

    def fake_run(argv, *, cwd, check):
        calls.append((argv, cwd, check))
        (tmp_path / "kernel.so").touch()

    monkeypatch.setattr(host_driver.subprocess, "run", fake_run)
    output = host_driver._compile(
        tmp_path, dict(host_driver._EXPECTED_BUILD), fake_torch, fake_torch_npu
    )

    assert output == tmp_path / "kernel.so"
    argv, cwd, check = calls[0]
    assert isinstance(argv, list)
    assert cwd == tmp_path and check is True
    assert argv[:7] == [
        "/opt/cann/bin/bisheng",
        "-x",
        "asc",
        "--npu-arch=dav-3510",
        "-shared",
        "-fPIC",
        "-std=c++17",
    ]
    assert "-ltorch_npu" in argv and "-ltorch" in argv and "-lc10" in argv
    assert not any(item in {"sh", "bash", "-c"} for item in argv)


def test_main_synchronizes_then_emits_one_exact_output_record(monkeypatch, capsys, tmp_path):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.tensor = lambda values, **unused: FakeTensor(values)
    fake_torch.npu = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        set_device=lambda device: events.append(("device", device)),
    )
    fake_torch.ops = SimpleNamespace(
        load_library=lambda path: events.append(("load", path)),
        a5kernels=SimpleNamespace(
            vector_add=lambda a, b: FakeTensor(
                [left + right for left, right in zip(a.values, b.values)]
            )
        ),
    )
    fake_torch_npu = ModuleType("torch_npu")
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(host_driver, "_compile", lambda *unused: tmp_path / "kernel.so")
    input_file = tmp_path / "input.json"
    input_file.write_text('{"input_a":[1.5,-2],"input_b":[2.25,1]}')
    monkeypatch.setattr(sys, "argv", ["host_driver.py", str(input_file)])
    monkeypatch.setattr(sys, "stdin", StringIO(""))
    monkeypatch.setenv("BZ_A5_PROFILE_PHYSICAL_DEVICE", "3")

    assert host_driver.main() == 0

    assert events == [("load", str(tmp_path / "kernel.so")), ("device", 3), "synchronize"]
    assert capsys.readouterr().out == "A5KERNEL_OUTPUT=[3.75,-1.0]\n"


def test_compile_fails_closed_without_bisheng(monkeypatch, tmp_path):
    monkeypatch.setattr(host_driver.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="bisheng compiler not found"):
        host_driver._compile(
            tmp_path,
            dict(host_driver._EXPECTED_BUILD),
            SimpleNamespace(),
            SimpleNamespace(),
        )


def test_host_rejects_oversized_vectors_before_dispatch():
    with pytest.raises(ValueError, match="cannot exceed 4096"):
        A5KernelRunner(SimpleNamespace()).prepare(RunRequest("ascend-c", length=4097))
