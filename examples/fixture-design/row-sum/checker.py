import importlib.util
import sys

import torch
import torch_npu


def check(run):
    for seed in (0, 13, 29):
        rng = torch.Generator().manual_seed(seed)
        raw = torch.randint(-6, 7, (3, 7), generator=rng).float()
        raw[:, -1] = torch.tensor([2.0, -3.0, 4.0])
        values = raw.npu()
        storage = torch.full((5,), -91.0, device=values.device)
        out = storage[1:4]
        result = run(values, out)
        torch.npu.synchronize()
        assert result.data_ptr() == out.data_ptr(), "output-storage: destination replaced"
        assert torch.equal(values.cpu(), raw), "input-values: input modified"
        assert storage[0].item() == storage[-1].item() == -91, "output-guards: outside write"
        expected = raw.sum(dim=1)
        got = out.cpu()
        if not torch.equal(expected, got):
            raise AssertionError(f"output-values: seed={seed} expected={expected.tolist()} got={got.tolist()}")


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("row_sum_control", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    check(module.run)
