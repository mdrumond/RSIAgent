"""Deterministic embeddings and independent arithmetic oracle for CLI acceptance."""
import hashlib
import importlib.util
from pathlib import Path
import sys


class Embeddings:
    model = 'deterministic-cpu'
    revision = '1'

    def embed(self, texts):
        return [[value / 255 for value in hashlib.sha256(text.encode()).digest()]
                for text in texts]


if __name__ == '__main__':
    if sys.argv[1] == 'baseline':
        from ascend_tools.kdb import KDB
        root = Path(sys.argv[2]).parent
        source = root / 'reference.py'
        source.write_text('def answer(): return 6 * 7\n')
        db = KDB(Path(sys.argv[2]), Embeddings())
        for kind in ('good', 'negative'):
            db.add_repo(str(root), 'fixture-' + kind, 'python', target='cpu',
                        kind=kind, paths=[source.name])
        db.close()
    elif sys.argv[1] == 'verify':
        spec = importlib.util.spec_from_file_location('candidate', Path(sys.argv[2]) / 'answer.py')
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        actual = candidate.answer()
        expected = sum(6 for _ in range(7))
        print({'actual': actual, 'expected': expected}, flush=True)
        assert actual == expected
