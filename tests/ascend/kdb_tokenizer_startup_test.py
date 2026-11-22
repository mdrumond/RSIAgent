"""Batch encoding in the KDB process respects explicit parallelism choices."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


@unittest.skipUnless(importlib.util.find_spec("tokenizers") and Path("/proc/self/task").is_dir(),
                     "requires native tokenizers and Linux process information")
class TokenizerStartupTests(unittest.TestCase):
    def test_service_tokenizes_without_allocating_a_host_sized_pool(self):
        script = textwrap.dedent("""
            import json, sys
            from pathlib import Path
            from unittest.mock import patch
            from tokenizers import Tokenizer, models
            from ascend_tools import kdb_service

            class Probe:
                def run(self, transport):
                    tokenizer = Tokenizer(models.WordLevel({"word": 0}, unk_token="word"))
                    tasks = Path("/proc/self/task")
                    before = len(list(tasks.iterdir()))
                    rows = tokenizer.encode_batch(["word"] * 8)
                    print(json.dumps([len(list(tasks.iterdir())) - before, [r.ids for r in rows]]))

            sys.argv = ["kdb_service", "--db", ":memory:"]
            with patch.object(kdb_service, "create_server", return_value=Probe()):
                kdb_service.main()
        """)
        env = {k: v for k, v in os.environ.items()
               if k not in ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS")}
        for settings, expected in (({}, 0), ({"RAYON_NUM_THREADS": "2"}, 2),
                                   ({"TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "2"}, 0)):
            with self.subTest(settings=settings):
                output = subprocess.check_output([sys.executable, "-c", script], env=env | settings)
                threads, rows = json.loads(output)
                self.assertEqual(threads, expected)
                self.assertEqual(rows, [[0]] * 8)
