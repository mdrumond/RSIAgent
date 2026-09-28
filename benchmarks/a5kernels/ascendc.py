"""Registry-owned sources for the AscendC A5 vector-add runtime."""

from __future__ import annotations

from importlib.resources import files

from benchmarks.a5kernels.fixtures import Fixture, Language
from benchmarks.a5kernels.protocol import SourceFile


_ASSET_PACKAGE = "benchmarks.a5kernels.ascendc_runtime"
_ASSET_NAMES = ("build.json", "host_driver.py", "kernel.cpp")


def ascendc_fixture() -> Fixture:
    """Return the complete, immutable staging set in stable path order."""

    root = files(_ASSET_PACKAGE)
    staged = tuple(
        SourceFile(name, root.joinpath(name).read_text(encoding="utf-8"))
        for name in _ASSET_NAMES
    )
    return Fixture(
        language=Language.ASCEND_C,
        files=staged,
        argv=("python", "host_driver.py", "input.json"),
        max_length=4096,
    )
