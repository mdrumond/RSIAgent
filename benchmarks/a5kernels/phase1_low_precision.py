"""Host-owned FP8 rounding and MX group32 scaling, without a Torch dependency."""

from bisect import bisect_left
from functools import lru_cache
import math

FP8_DTYPES = {"float8_e4m3fn", "float8_e5m2"}
MX_DTYPES = {"mxfp8_e4m3fn", "mxfp8_e5m2"}


def storage_dtype(dtype: str) -> str:
    return dtype.replace("mxfp8_", "float8_")


@lru_cache(maxsize=2)
def _positive_values(mantissa, bias, last):
    return tuple(
        ((code % (1 << mantissa)) / (1 << mantissa)
         * 2.0 ** (1 - bias)) if code < (1 << mantissa) else
        (1 + (code % (1 << mantissa)) / (1 << mantissa))
        * 2.0 ** ((code >> mantissa) - bias)
        for code in range(last + 1)
    )

def quantize_fp8(value: float, dtype: str) -> float:
    """IEEE round-to-nearest-even, including subnormals and format overflow."""
    e4 = storage_dtype(dtype) == "float8_e4m3fn"
    mantissa, bias, last = (3, 7, 126) if e4 else (2, 15, 123)
    magnitude = abs(float(value))
    if math.isnan(magnitude):
        return math.nan
    # Positive encoding order is numerical order; encoding parity selects ties.
    values = _positive_values(mantissa, bias, last)
    if magnitude >= (464.0 if e4 else 61440.0):
        # E4M3FN's midpoint belongs to its even maximum finite encoding.
        if e4 and magnitude == 464.0:
            return math.copysign(448.0, value)
        return math.nan if e4 else math.copysign(math.inf, value)
    index = bisect_left(values, magnitude)
    if index == len(values):
        index -= 1
    elif index and (magnitude - values[index - 1] < values[index] - magnitude
                   or (magnitude - values[index - 1] == values[index] - magnitude
                       and index % 2)):
        index -= 1
    return math.copysign(values[index], value)


def mx_quantize(shape, storage, dtype, operand):
    """Return FP8 values, E8M0 bytes and dequantized values in logical order.

    A scales are row-major (M,G); B scales are physically (N,G), but have
    logical shape (G,N) with ColMajorMxScaleB. G is padded to an even count.
    This fixes the supported layout pair to row-major A / column-major B.
    """
    rows, cols = shape
    k = cols if operand == 0 else rows
    outer = rows if operand == 0 else cols
    groups = (k + 31) // 32
    padded = (groups + 1) // 2 * 2
    quantized, dequantized = list(storage), list(storage)
    scales = []
    emax, maximum = (8, 448.0) if storage_dtype(dtype) == "float8_e4m3fn" else (15, 57344.0)
    for row in range(outer):
        for group in range(groups):
            indexes = [row * cols + inner if operand == 0 else inner * cols + row
                       for inner in range(group * 32, min(k, (group + 1) * 32))]
            peak = max(abs(storage[index]) for index in indexes)
            exponent = 0 if peak < 1e-30 else max(-127, min(127, math.floor(math.log2(peak)) - emax))
            scale = 2.0 ** exponent
            scales.append(exponent + 127)
            for index in indexes:
                quantized[index] = quantize_fp8(max(-maximum, min(maximum, storage[index] / scale)), dtype)
                dequantized[index] = quantized[index] * scale
        scales.extend([127] * (padded - groups))
    return quantized, {"shape": [outer, padded], "storage": scales}, dequantized
