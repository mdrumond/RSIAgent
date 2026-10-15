"""Stage a bounded three-input native probe; no remote access or builds here."""

import argparse
import hashlib
import json
from pathlib import Path

from .native_candidate import NativeCandidatePlan, NativeTensor


SOURCE = '''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void sum_three(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR out) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::QuePosition::VECIN> bx, by, bz;
  AscendC::TBuf<AscendC::QuePosition::VECOUT> bo;
  pipe.InitBuffer(bx, 128); pipe.InitBuffer(by, 128);
  pipe.InitBuffer(bz, 128); pipe.InitBuffer(bo, 128);
  AscendC::GlobalTensor<float> gx, gy, gz, go;
  gx.SetGlobalBuffer((__gm__ float*)x, 32);
  gy.SetGlobalBuffer((__gm__ float*)y, 32);
  gz.SetGlobalBuffer((__gm__ float*)z, 32);
  go.SetGlobalBuffer((__gm__ float*)out, 32);
  auto lx=bx.Get<float>(); auto ly=by.Get<float>();
  auto lz=bz.Get<float>(); auto lo=bo.Get<float>();
  AscendC::DataCopy(lx,gx,32); AscendC::DataCopy(ly,gy,32); AscendC::DataCopy(lz,gz,32);
  auto loaded=pipe.FetchEventID<AscendC::HardEvent::MTE2_V>();
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loaded);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loaded);
  AscendC::Add(lo,lx,ly,32);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Add(lo,lo,lz,32);
  auto computed=pipe.FetchEventID<AscendC::HardEvent::V_MTE3>();
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(computed);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(computed);
  AscendC::DataCopy(go,lo,32);
}
'''


def smoke_plan(*, target, arch, profile="bz-a3-1"):
    inputs = tuple(NativeTensor(name, (2, 16), "float32", tuple(float(i + offset) for i in range(32))) for name, offset in (("x", 0), ("y", 2), ("z", -3)))
    return NativeCandidatePlan(
        SOURCE, "sum_three", inputs, NativeTensor("out", (32,), "float32"),
        "native-smoke", "three-input-fp32", "sum-three", profile, target, arch,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--profile", choices=("bz-a3-1", "bz-a3-2", "gz-a3"), default="bz-a3-1")
    args = parser.parse_args()
    plan = smoke_plan(target=args.target, arch=args.arch, profile=args.profile)
    args.stage.mkdir(parents=True, exist_ok=False)
    for item in plan.files:
        (args.stage / item.relative_path).write_text(item.content)
    (args.stage / "input.json").write_text(json.dumps(plan.input_payload))
    manifest = dict(archive_schema="rsi-a3-candidate-v1", execution_id=plan.execution_id,
                    source_fingerprint=plan.source_fingerprint,
                    files={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in args.stage.iterdir() if path.is_file()})
    (args.stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    # This is a separate host oracle, never supplied to the candidate runtime.
    expected = [sum(values) for values in zip(*(t.values for t in plan.inputs))]
    print(json.dumps({"execution_id": plan.execution_id, "expected_host_output": expected}))


if __name__ == "__main__":
    main()
