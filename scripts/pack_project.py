"""Export one project, or the whole campaign when --project is omitted."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascend_tools.storage_package import pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('dest')
    parser.add_argument('--project')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(pack(args.root, args.dest, args.project, not args.allow_incomplete)))


if __name__ == '__main__':
    main()
