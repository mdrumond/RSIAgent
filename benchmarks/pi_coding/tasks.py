"""Registered coding tasks and independent, host-owned numerical references."""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


TASK_IDS = ("vector-add", "matmul-add")
MATMUL_SHAPES = ((64, 64, 64), (129, 193, 257), (257, 321, 193))  # M, N, K
TILE = (64, 64, 32)
FIXTURES = Path(__file__).with_name("fixtures")


def task_contract(task):
    if task == "vector-add":
        return {"task": task, "lengths": [1, 7, 32, 257, 4096], "dtype": "float32"}
    if task == "matmul-add":
        return {"task": task, "equation": "D=A@B+C", "shapes_mnk": [list(s) for s in MATMUL_SHAPES],
                "input_ab": "float16", "input_c_output": "float32", "tile_mnk": list(TILE),
                "rtol": 1e-4, "atol": 1e-4}
    raise ValueError("unknown coding task")


def contract_digest(task):
    return hashlib.sha256(json.dumps(task_contract(task), sort_keys=True).encode()).hexdigest()


def task_description(task):
    if task == "vector-add":
        return "Repair the compile error in the Ascend C vector-add kernel. Preserve its operator interface and correct float32 addition for lengths 1 through 4096, including unaligned lengths."
    task_contract(task)  # Reject unregistered tasks before any provider calls.
    return """Write an Ascend C matmul-plus-add device kernel from the TODO starter.
Compute D[M,N] = A[M,K] @ B[K,N] + C[M,N], row-major and contiguous.
A/B are float16; C/output and accumulation are float32. The fixed test cases
(M,N,K) are (64,64,64), (129,193,257), and (257,321,193).
Use 64x64 output tiles and a loop over K in chunks of 32. The trusted host wrapper
launches ceil(M/64)*ceil(N/64) blocks, one output tile per block. Decode GetBlockIdx()
as tile_row = block / ceil(N/64), tile_col = block % ceil(N/64). The two larger
cases have 12 and 30 output tiles, and 9 and 7 K chunks respectively. Handle partial
M, N, and K tiles without out-of-bounds accesses; keep working buffers tile-sized.
Implement exactly the supplied MatmulAddTile signature. You edit only the device
kernel.cpp, compiled into a trusted Torch binding supplied by the host. No ATen,
torch, host-side matmul, CPU fallback, or precomputed answers in candidate code.
Ascend C cube or vector/scalar instructions are allowed; this task checks numerical
correctness, not peak throughput. Use Ascend C APIs available in CANN 9.1 on 910B.
"""


def matmul_starter():
    return (FIXTURES / "matmul_add_kernel.cpp").read_text()


def matmul_inputs(m, n, k, seed):
    a = [[((i * 17 + p * 13 + seed * 3) % 31 - 15) / 8 for p in range(k)] for i in range(m)]
    b = [[((p * 11 + j * 7 + seed * 5) % 29 - 14) / 8 for j in range(n)] for p in range(k)]
    c = [[((i * 3 + j * 5 + seed) % 17 - 8) / 4 for j in range(n)] for i in range(m)]
    return a, b, c


@lru_cache(maxsize=1)
def matmul_references():
    # Dyadic input fractions are exactly representable in FP16 and in Python float.
    # Independent CPU reference: never accept a candidate-reported error metric.
    cases = []
    for seed, (m, n, k) in enumerate(MATMUL_SHAPES, 1):
        a, b, c = matmul_inputs(m, n, k, seed)
        columns = list(zip(*b))
        cases.append(tuple(tuple(sum(x * y for x, y in zip(row, col)) + c[i][j]
                                 for j, col in enumerate(columns)) for i, row in enumerate(a)))
    return tuple(cases)


def verify_matmul_outputs(rows):
    if not isinstance(rows, list) or len(rows) != len(MATMUL_SHAPES):
        return False
    for shape, actual, expected in zip(MATMUL_SHAPES, rows, matmul_references()):
        m, n, _k = shape
        if not isinstance(actual, list) or len(actual) != m:
            return False
        for row, reference in zip(actual, expected):
            if not isinstance(row, list) or len(row) != n:
                return False
            if any(type(v) not in (int, float) or not math.isfinite(v) or
                   not math.isclose(v, r, abs_tol=1e-4, rel_tol=1e-4)
                   for v, r in zip(row, reference)):
                return False
    return True


def matmul_feedback(rows):
    feedback = []
    for index, ((m, n, k), expected) in enumerate(zip(MATMUL_SHAPES, matmul_references())):
        case = {"shape_mnk": [m, n, k], "output_tiles": ((m + 63) // 64) * ((n + 63) // 64),
                "k_chunks": (k + 31) // 32, "passed": False}
        feedback.append(case)
        if not isinstance(rows, list) or index >= len(rows) or not isinstance(rows[index], list) or len(rows[index]) != m:
            case["reason"] = "missing output or incorrect row count"
            continue
        actual = rows[index]
        if any(not isinstance(row, list) or len(row) != n for row in actual):
            case["reason"] = "incorrect column count"
            continue
        case["passed"] = True
        maximum = 0.0
        for i, row in enumerate(actual):
            for j, value in enumerate(row):
                reference = expected[i][j]
                finite = type(value) in (int, float) and math.isfinite(value)
                if finite:
                    maximum = max(maximum, abs(value - reference))
                if not finite or not math.isclose(value, reference, abs_tol=1e-4, rel_tol=1e-4):
                    case["passed"] = False
                    if "first_mismatch" not in case:
                        case["first_mismatch"] = {"row": i, "column": j, "expected": reference,
                                                  "actual": value if finite else "nonfinite-or-invalid"}
        case["max_abs_error"] = maximum
    return feedback


MATMUL_RUN = '''        rows = []
        for seed, (m, n, k) in enumerate(((64,64,64), (129,193,257), (257,321,193)), 1):
            result["active_case"] = {"shape_mnk": [m, n, k], "seed": seed}
            a = torch.tensor([[((i*17+p*13+seed*3)%31-15)/8 for p in range(k)] for i in range(m)], dtype=torch.float16, device="npu:0")
            b = torch.tensor([[((p*11+j*7+seed*5)%29-14)/8 for j in range(n)] for p in range(k)], dtype=torch.float16, device="npu:0")
            c = torch.tensor([[((i*3+j*5+seed)%17-8)/4 for j in range(n)] for i in range(m)], dtype=torch.float32, device="npu:0")
            output = torch.ops.rsi_pi_matmul.matmul_add(a, b, c)
            torch.npu.synchronize()
            rows.append(output.cpu().tolist())
'''
