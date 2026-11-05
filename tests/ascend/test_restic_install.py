"""Installer resolution and verified atomic download behavior without networking."""
import bz2
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_tools import restic_install as installer


class ResticInstall(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / 'tools' / 'restic'
        self.enterContext(patch.object(installer, 'cache_path', return_value=self.target))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(installer.shutil, 'which', return_value=None))

    def test_explicit_override_and_cached_resolution_do_not_download(self):
        with patch.dict(os.environ, {'RSI_RESTIC': '/chosen/restic'}):
            self.assertEqual(installer.resolve_restic(), '/chosen/restic')
        self.target.parent.mkdir()
        self.target.write_bytes(b'existing')
        self.assertEqual(installer.resolve_restic(), str(self.target))

    def test_missing_resolution_is_actionable_and_does_not_install(self):
        with self.assertRaises(RuntimeError):
            installer.resolve_restic()
        self.assertFalse(self.target.exists())

    def test_checked_download_publishes_executable_atomically(self):
        payload = bz2.compress(b'#!/bin/sh\nexit 0\n')
        with patch.object(installer.platform, 'system', return_value='Linux'), \
             patch.object(installer.platform, 'machine', return_value='x86_64'), \
             patch.dict(installer.HASHES, {'amd64': hashlib.sha256(payload).hexdigest()}), \
             patch.object(installer.urllib.request, 'urlopen', return_value=io.BytesIO(payload)), \
             patch.object(installer.subprocess, 'run') as run:
            self.assertEqual(installer.install(), str(self.target))
        self.assertEqual(self.target.read_bytes(), b'#!/bin/sh\nexit 0\n')
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o700)
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_bad_checksum_never_publishes(self):
        with patch.object(installer.platform, 'system', return_value='Linux'), \
             patch.object(installer.platform, 'machine', return_value='aarch64'), \
             patch.object(installer.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong')):
            with self.assertRaises(RuntimeError):
                installer.install()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
