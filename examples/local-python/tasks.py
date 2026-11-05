"""Small CPU workloads and an independent oracle; no third-party dependencies."""
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

BASELINE = '''def reduce_tiles(values, tile=4):
    return sum(sum(values[i:i + tile]) for i in range(0, len(values) - tile + 1, tile))

def parse_ints(text):
    return [int(part) for part in text.split(",")]
'''
REPAIR = '''def reduce_tiles(values, tile=4):
    return sum(sum(values[i:i + tile]) for i in range(0, len(values), tile))

def parse_ints(text):
    return [] if not text.strip() else [int(part.strip()) for part in text.split(",")]
'''
REGRESSION = REPAIR.replace('int(part.strip())', 'abs(int(part.strip()))')


def verify(kind, source):
    spec = importlib.util.spec_from_file_location('candidate', Path(source) / 'answer.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    if kind == 'reduction':
        for size in (0, 1, 3, 4, 5, 9, 17):
            values = [((i * 7) % 13) - 6 for i in range(size)]
            for tile in (1, 4, 7):
                actual = candidate.reduce_tiles(values, tile)
                expected = sum(values)
                assert actual == expected, (size, tile, actual, expected)
    elif kind == 'parser':
        for text, expected in (('', []), ('  ', []), ('1', [1]),
                               (' -2, 0, 3 ', [-2, 0, 3])):
            actual = candidate.parse_ints(text)
            assert actual == expected, (text, actual, expected)
        for text in ('1,,2', 'one', '1,'):
            try:
                candidate.parse_ints(text)
            except ValueError:
                continue
            raise AssertionError(('invalid input accepted', text))
    else:
        raise ValueError(kind)
    print(json.dumps({'kind': kind, 'passed': True}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        verify(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'source':
        print({'baseline': BASELINE, 'repair': REPAIR, 'regression': REGRESSION}[sys.argv[2]], end='')
    else:
        raise ValueError('Use verify KIND SOURCE or source baseline|repair|regression')
