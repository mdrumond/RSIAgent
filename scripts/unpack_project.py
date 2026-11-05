"""Restore a package into a new, paused destination; never execute its contents."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascend_tools.storage_package import unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package')
    parser.add_argument('dest')
    args = parser.parse_args()
    print(json.dumps(unpack(args.package, args.dest)))


if __name__ == '__main__':
    main()
