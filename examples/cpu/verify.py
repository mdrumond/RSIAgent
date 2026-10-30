"""Independent decimal reference for a small numeric programming project."""
from decimal import Decimal
import math

from solution import stable_sum


def main():
    cases = ([], [1., 2., 3.], [1e16, 1., -1e16], [0.5, -0.25, 0.125])
    for values in cases:
        expected = float(sum((Decimal.from_float(x) for x in values), Decimal(0)))
        if not math.isclose(stable_sum(values), expected, rel_tol=1e-15, abs_tol=0.):
            raise AssertionError(f"numeric mismatch for {values}")
    print("CPU numeric verification passed")


if __name__ == "__main__":
    main()
