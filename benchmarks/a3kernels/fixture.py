"""Registry-owned A3 Ascend C runtime sources."""

from importlib.resources import files

from benchmarks.a3kernels.protocol import SourceFile


_PACKAGE = "benchmarks.a3kernels.ascendc_runtime"
_NAMES = ("build.json", "host_driver.py", "kernel.cpp")


def source_files() -> tuple[SourceFile, ...]:
    root = files(_PACKAGE)
    return tuple(
        SourceFile(name, root.joinpath(name).read_text(encoding="utf-8"))
        for name in _NAMES
    )
