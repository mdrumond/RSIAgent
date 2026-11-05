"""Small synthetic artifacts only; no campaign fixtures or dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

from ascend_tools.storage_archive import (
    compress_file, finalize_source, iter_artifact_bytes, restore_file,
)
from ascend_tools.log_walk import open_artifact_bytes, walk_records


class ArchivesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.jsonl'
        self.archive = self.root / 'archive'
        self.source.write_bytes(b''.join(bytes([i]) * 100 for i in range(5)))

    def test_interrupt_resume_reuses_verified_chunks(self):
        # Exit the interpreter between checkpoints, not a mocked compressor.
        code = '''
import os, sys
from ascend_tools.storage_archive import compress_file
compress_file(sys.argv[1], sys.argv[2], chunk_bytes=100,
              on_chunk=lambda manifest: os._exit(23))
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.source), str(self.archive)])
        self.assertEqual(result.returncode, 23)
        progress = json.loads(next(self.archive.glob('*.progress.json')).read_text())
        self.assertEqual(len(progress['chunks']), 1)
        chunk = self.archive / progress['chunks'][0]['file']
        before = chunk.stat().st_mtime_ns
        # An interrupted temporary write is ignored, not considered completed.
        (self.archive / '.chunk-interrupted').write_bytes(b'partial')
        manifest = compress_file(self.source, self.archive, chunk_bytes=100)
        self.assertEqual(chunk.stat().st_mtime_ns, before)
        self.assertEqual(b''.join(iter_artifact_bytes(manifest)), self.source.read_bytes())
        restored = self.root / 'restored'
        restore_file(manifest, restored)
        self.assertEqual(restored.read_bytes(), self.source.read_bytes())
        with self.assertRaises(FileExistsError):
            restore_file(manifest, restored)
        self.assertTrue(self.source.exists())
        finalize_source(manifest)
        self.assertFalse(self.source.exists())

    def test_corrupt_completed_chunk_is_repaired_on_resume(self):
        manifest = compress_file(self.source, self.archive, chunk_bytes=80)
        chunk = self.archive / manifest['chunks'][0]['file']
        chunk.write_bytes(b'broken')
        destination = self.root / 'destination'
        with self.assertRaises((OSError, ValueError, EOFError)):
            restore_file(manifest, destination)
        self.assertFalse(destination.exists())
        repaired = compress_file(self.source, self.archive, chunk_bytes=80)
        self.assertEqual(b''.join(iter_artifact_bytes(repaired)), self.source.read_bytes())

    def test_damaged_deflate_payload_is_retried(self):
        manifest = compress_file(self.source, self.archive, chunk_bytes=100)
        chunk = self.archive / manifest['chunks'][0]['file']
        payload = bytearray(chunk.read_bytes())
        payload[10] = 0xff  # invalid deflate block after the gzip header
        chunk.write_bytes(payload)
        with self.assertRaises((OSError, EOFError, ValueError, zlib.error)):
            list(iter_artifact_bytes(manifest))
        repaired = compress_file(self.source, self.archive, chunk_bytes=100)
        self.assertEqual(b''.join(iter_artifact_bytes(repaired)), self.source.read_bytes())

    def test_changed_source_is_not_removed_or_resumed(self):
        manifest = compress_file(self.source, self.archive, chunk_bytes=100)
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            finalize_source(manifest)
        with self.assertRaises(ValueError):
            compress_file(self.source, self.archive, chunk_bytes=100)
        self.assertEqual(self.source.read_bytes(), b'changed')

    def test_source_changed_between_chunks_aborts(self):
        def mutate(manifest):
            with self.source.open('ab') as out:
                out.write(b'new')
        with self.assertRaises(ValueError):
            compress_file(self.source, self.archive, chunk_bytes=100, on_chunk=mutate)
        self.assertFalse(list(self.archive.glob('*.manifest.json')))
        self.assertTrue(self.source.exists())

    def test_finalize_requires_intact_archive_and_chunk_limit(self):
        with self.assertRaises(ValueError):
            compress_file(self.source, self.archive, chunk_bytes=32 * 1024 * 1024 + 1)
        manifest = compress_file(self.source, self.archive, chunk_bytes=100)
        (self.archive / manifest['chunks'][-1]['file']).write_bytes(b'truncated')
        with self.assertRaises((OSError, EOFError, ValueError)):
            finalize_source(manifest)
        self.assertTrue(self.source.exists())

    def test_empty_and_expected_identity(self):
        self.source.write_bytes(b'')
        with self.assertRaises(ValueError):
            compress_file(self.source, self.archive, expected_identity={})
        manifest = compress_file(self.source, self.archive)
        self.assertEqual(b''.join(iter_artifact_bytes(manifest)), b'')
        restore_file(manifest, self.root / 'empty')
        self.assertEqual((self.root / 'empty').stat().st_size, 0)

    def test_native_records_filter_cursors_and_raw_references(self):
        records = [dict(type='session', id='abc', version=3),
                   dict(type='message', message=dict(role='assistant', content=[
                       dict(type='text', text='hello')]))]
        data = b'\n'.join(json.dumps(x).encode() for x in records)
        self.source.write_bytes(data)
        plain = list(walk_records(self.source))
        self.assertEqual([x['record'] for x in plain], records)
        cursor = plain[0]['metadata']['next_cursor']
        manifest = compress_file(self.source, self.archive, chunk_bytes=7)
        archived = list(walk_records(manifest))
        self.assertEqual([x['record'] for x in archived], records)
        self.assertEqual(b''.join(open_artifact_bytes(archived[0]['metadata']['raw_ref']['artifact'])), data)
        selected = list(walk_records(manifest, cursor=cursor,
                        predicate=lambda record, meta: record['type'] == 'message'))
        self.assertEqual(selected, archived[1:])
        self.assertEqual(b''.join(open_artifact_bytes(manifest)), data)

    def test_oversized_and_invalid_records_are_referenced(self):
        self.source.write_bytes(b'x' * 300000 + b'\nnot json\n{"type":"session"}\n')
        items = list(walk_records(self.source, record_bytes=32))
        self.assertEqual([x['metadata']['status'] for x in items],
                         ['oversized', 'invalid', 'ok'])
        self.assertIsNone(items[0]['record'])
        ref = items[0]['metadata']['raw_ref']
        with open(ref['artifact'], 'rb') as stream:
            stream.seek(ref['offset'])
            self.assertEqual(stream.read(ref['length']), b'x' * 300000 + b'\n')
        self.assertEqual(items[-1]['record'], {'type': 'session'})

    def test_streaming_rss_small_fixture(self):
        # A fresh process measures RSS rather than inheriting this runner's peak.
        code = '''
import resource, sys
from pathlib import Path
from ascend_tools.storage_archive import compress_file, iter_artifact_bytes
from ascend_tools.log_walk import walk_records
p = Path(sys.argv[1])
with p.open('wb') as out:
    for _ in range(256): out.write(b'x' * 65536)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
m = compress_file(p, sys.argv[2], chunk_bytes=262144)
assert sum(map(len, iter_artifact_bytes(m))) == 16 * 1024 * 1024
assert list(walk_records(m, record_bytes=1024))[0]['metadata']['status'] == 'oversized'
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 12 * 1024
'''
        subprocess.run([sys.executable, '-c', code, str(self.source), str(self.archive)], check=True)


if __name__ == '__main__':
    unittest.main()
