"""Require a live launch of an artifact compiled from or through the submitted module."""
import json
import runpy
import sys
from pathlib import Path

import catlass.tla as tla

check, source = sys.argv[1:]
submitted = Path(source).resolve()
counts = [0, 0]
native = tla.compile


def from_submission(fn):
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        code = getattr(fn, "__code__", None)
        if code is not None and Path(code.co_filename).resolve() == submitted:
            return True
        fn = next((getattr(fn, name, None) for name in
                   ("__wrapped__", "fn", "func", "py_func", "_func")
                   if callable(getattr(fn, name, None))), None)
    return False


def compile_spy(*args, **kwargs):
    frame = sys._getframe(1)
    submitted_caller = False
    while frame:
        if Path(frame.f_code.co_filename).resolve() == submitted:
            submitted_caller = True
            break
        frame = frame.f_back
    op = native(*args, **kwargs)
    if not submitted_caller and not any(from_submission(v) for v in (*args, *kwargs.values())):
        return op
    counts[0] += 1

    def launch(*args, **kwargs):
        result = op(*args, **kwargs)
        counts[1] += 1
        return result
    return launch


tla.compile = compile_spy
sys.argv = [check, source]
try:
    runpy.run_path(check, run_name="__main__")
except SystemExit as exit_status:
    if exit_status.code not in (None, 0):
        if all(counts):
            print("VERIFIED_NATIVE_ROUTE", *counts, flush=True)
        raise
    # A successful CLI exit must pass the same route check as a normal return.
except BaseException as failure:
    trace = failure.__traceback__
    while trace and trace.tb_next:
        trace = trace.tb_next
    if isinstance(failure, AssertionError) and trace and Path(trace.tb_frame.f_code.co_filename).resolve() == Path(check).resolve():
        check_id, separator, details = str(failure).partition(": ")
        if separator and details.strip():
            print("VERIFIED_CHECK_FAILURE", json.dumps({"id": check_id, "details": details}), flush=True)
    if all(counts):
        print("VERIFIED_NATIVE_ROUTE", *counts, flush=True)
    raise
assert all(counts), f"no native Catlass launch of submitted source: {counts}"
print("VERIFIED_NATIVE_ROUTE", *counts)
