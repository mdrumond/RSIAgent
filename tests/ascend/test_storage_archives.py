"""Behavioral tests using actual Restic and synthetic, task-owned files only."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from ascend_tools import storage_archive as archive
from ascend_tools.log_walk import walk_records


class ResticArchives(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repository'
        self.source = self.root / 'native.jsonl'
        self.records = [dict(type='session', id='native-session', version=3),
                        dict(type='message', id='native-message', parentId=None,
                             message=dict(role='assistant', content=[dict(type='text', text='hello')])),
                        dict(type='custom', customType='opaque', data={'preserved': [1, False, None]})]
        self.source.write_bytes(b''.join(json.dumps(x).encode() + b'\n' for x in self.records))

    def test_batch_backup_restore_dedup_and_no_overwrite(self):
        paths = [self.source]
        for i in range(80):
            path = self.root / (f'file-{i}.txt' if i else 'file-[literal]\nname')
            path.write_bytes(self.source.read_bytes())
            paths.append(path)
        with patch.object(archive, '_run', wraps=archive._run) as run:
            batch = archive.archive_files(paths, self.repo)
            self.assertEqual(sum(c.args[1] == 'backup' for c in run.call_args_list), 1)
        self.assertEqual(len(batch['files']), len(paths))
        first = json.loads(archive._run(self.repo, 'stats', batch['snapshot'], '--mode', 'raw-data', '--json'))
        summaries = []
        real = archive._run
        def capture(repository, *args):
            result = real(repository, *args)
            if args[0] == 'backup':
                summaries.extend(json.loads(line) for line in result.splitlines())
            return result
        with patch.object(archive, '_run', side_effect=capture):
            archive.archive_files(paths, self.repo)
        self.assertGreater(first['total_size'], 0)
        self.assertEqual(summaries[-1]['data_blobs'], 0)
        archive._run(self.repo, 'check', '--read-data')
        output = self.root / 'out'
        output.mkdir()
        selected = [dict(m, logical_path=f'renamed-{i}') for i, m in enumerate(batch['files'][:5])]
        selected.append(dict(selected[0], logical_path='duplicate-source-alias'))
        with patch.object(archive, '_run', wraps=archive._run) as run:
            restored = archive.restore_files(selected, lambda m: output / m['logical_path'])
            self.assertEqual(sum(c.args[1] == 'restore' for c in run.call_args_list), 1)
        for name, original in zip(restored, paths[:5] + [paths[0]]):
            self.assertEqual(Path(name).read_bytes(), original.read_bytes())
        self.assertEqual(len(list(output.iterdir())), 6)
        with self.assertRaises(FileExistsError):
            archive.restore_files(selected, lambda m: output / m['logical_path'])
        self.assertEqual(len(list(output.iterdir())), 6)
        self.assertEqual(self.repo.stat().st_mode & 0o777, 0o700)

    def test_verified_atomic_restore_and_explicit_finalize(self):
        manifest = archive.compress_file(self.source, self.repo)
        self.assertEqual(json.loads(Path(manifest['manifest_path']).read_text()), manifest)
        self.assertTrue(self.source.exists())
        destination = self.root / 'restored'
        broken = dict(manifest, sha256='0' * 64)
        with self.assertRaises(ValueError):
            archive.restore_file(broken, destination)
        self.assertFalse(destination.exists())
        with self.assertRaises(ValueError):
            archive.finalize_source(broken)
        self.assertTrue(self.source.exists())
        self.assertFalse(list(self.root.glob('.restore-*')))
        archive.restore_file(manifest, destination)
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        archive.finalize_source(manifest)
        self.assertFalse(self.source.exists())
        self.assertEqual(b''.join(archive.iter_artifact_bytes(manifest)), destination.read_bytes())

    def test_source_changed_during_backup_and_before_finalize(self):
        real = archive._run
        def change(repository, *args):
            result = real(repository, *args)
            if args[0] == 'backup':
                self.source.write_text('changed after backup')
            return result
        with patch.object(archive, '_run', side_effect=change):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                archive.archive_files([self.source], self.repo)
        manifest = archive.compress_file(self.source, self.repo)
        self.source.write_text('changed before publication')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            archive.finalize_source(manifest)
        self.assertEqual(self.source.read_text(), 'changed before publication')

    def test_native_pi_records_plain_and_restic_with_cursors(self):
        manifest = archive.compress_file(self.source, self.repo)
        for artifact in [self.source, manifest, {'manifest': manifest['manifest_path']}]:
            records = list(walk_records(artifact))
            self.assertEqual([x['record'] for x in records], self.records)
            cursor = records[0]['metadata']['next_cursor']
            resumed = list(walk_records(artifact, cursor=cursor))
            self.assertEqual([x['record'] for x in resumed], self.records[1:])
            filtered = list(walk_records(artifact, predicate=lambda r, m: r['type'] == 'custom'))
            self.assertEqual([x['record'] for x in filtered], self.records[2:])
        with self.assertRaisesRegex(ValueError, 'boundary'):
            list(walk_records(manifest, cursor=2))
        broken = dict(manifest, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'integrity'):
            list(walk_records(broken, cursor=cursor))
        self.source.write_bytes(b'not JSON\n' + b'x' * 40 + b'\n{}\n')
        statuses = list(walk_records(self.source, record_bytes=16))
        self.assertEqual([r['metadata']['status'] for r in statuses], ['invalid', 'oversized', 'ok'])
        self.assertEqual(statuses[1]['metadata']['raw_ref']['length'], 41)

    def test_dump_size_digest_and_early_close(self):
        self.source.write_bytes(os.urandom(256 * 1024))
        manifest = archive.compress_file(self.source, self.repo)
        iterator = archive.iter_artifact_bytes(manifest)
        self.assertLessEqual(len(next(iterator)), 64 * 1024)
        iterator.close()
        # Closing a partially consumed dump must release both process and lock.
        self.assertEqual(hashlib.sha256(b''.join(archive.iter_artifact_bytes(manifest))).hexdigest(), manifest['sha256'])
        with self.assertRaisesRegex(ValueError, 'integrity'):
            list(archive.iter_artifact_bytes(dict(manifest, size=1)))

    def test_interrupted_actual_backup_reruns_and_checks(self):
        # Kill the actual backup subprocess after it acquires its native lock.
        # The Python caller reaps it, leaving a genuinely stale Restic lock.
        with self.source.open('wb') as out:
            for _ in range(32):
                out.write(os.urandom(1024 * 1024))
        original = archive._run
        interrupted = []
        def interrupt(repository, *args):
            if args[0] != 'backup':
                return original(repository, *args)
            command, env = archive._command(repository, *args)
            with subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) as process:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if any((repository / 'locks').iterdir()):
                        # Allow real backup work before abrupt death.
                        time.sleep(.05)
                        process.kill()
                        break
                    if process.poll() is not None:
                        self.fail('backup completed before interruption')
                    time.sleep(.005)
                else:
                    process.kill()
                    self.fail('backup did not acquire lock')
                interrupted.append(process.wait(timeout=10))
            raise subprocess.CalledProcessError(interrupted[-1], command)
        with patch.object(archive, '_run', side_effect=interrupt):
            with self.assertRaises(subprocess.CalledProcessError):
                archive.archive_files([self.source], self.repo)
        self.assertEqual(interrupted, [-signal.SIGKILL])
        self.assertTrue(list((self.repo / 'locks').iterdir()))
        manifest = archive.archive_files([self.source], self.repo)['files'][0]
        archive._run(self.repo, 'check', '--read-data')
        restored = self.root / 'resumed'
        archive.restore_file(manifest, restored)
        self.assertEqual(archive._sha(restored), archive._sha(self.source))
        self.assertFalse(list((self.repo / 'locks').iterdir()))

    def test_only_local_repository_and_regular_inputs(self):
        with self.assertRaises(ValueError):
            archive.archive_files([self.source], 'rest:http://localhost/repository')
        with self.assertRaises(ValueError):
            archive.archive_files([self.root], self.repo)
        with self.assertRaises(ValueError):
            archive.archive_files([], self.repo)


if __name__ == '__main__':
    unittest.main()
