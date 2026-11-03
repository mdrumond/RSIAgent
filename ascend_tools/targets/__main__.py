"""Execute through the configured environment's ordinary command tool."""
import argparse
import json
from .runner import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('cpu', 'a3-triton', 'a5-catlass'), required=True)
    parser.add_argument('--physical-device', type=int)
    parser.add_argument('--mode', choices=('compile', 'test', 'performance'), required=True)
    for name in ('candidate', 'cases', 'cases-sha256', 'runtime', 'runtime-sha256', 'source-sha256', 'artifacts'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if args.target != 'cpu' and (args.physical_device is None or args.physical_device < 0):
        parser.error('--physical-device must explicitly select a nonnegative physical ID for device targets')
    report = verify(args)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
