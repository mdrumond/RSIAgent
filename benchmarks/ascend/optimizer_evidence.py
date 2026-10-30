"""Authenticate committed optimizer-backed Optimizer task learning and package its evidence."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from optimizer.jobs import JobStore, canonical, sha
from optimizer.memory import verify_snapshot




def _artifact(store, key, digest):
    # The caller has authenticated this job and its event chain. Do not rehash
    # the large immutable submission once per compact profiler CSV.
    data = store.artifact_path(key, digest).read_bytes()
    if sha(data) != digest:
        raise ValueError("Optimizer task artifact digest mismatch")
    return data


def verify_task_submission(submission, task_submission):
    if (any(submission.get(k) != task_submission.get(k) for k in ('source', 'entrypoint'))
            or submission['signature'] != {'signatures': task_submission['signatures']}
            or submission['init_args'] != task_submission.get('init_args', [])
            or submission['init_kwargs'] != task_submission.get('init_kwargs', {})):
        raise ValueError('Optimizer task task differs from frozen optimizer submission')


def verify_attempt_result(store, result, *, attempt_id, task_id, task_submission):
    """Authenticate a retained attempt verdict against native execution evidence."""
    key = result['job_id']
    verify_task_submission(store.submission(key), task_submission)
    events = store.events(key)
    if not any(e['kind'] == 'task-attempt-result' and e['payload'] == result for e in events):
        raise ValueError('Optimizer task attempt result missing from optimizer event chain')
    body = json.loads(_artifact(store, key, result['result_artifact']['sha256']))
    if body != {k: v for k, v in result.items() if k != 'result_artifact'}:
        raise ValueError('Optimizer task result artifact mismatch')
    if result['attempt_id'] != attempt_id or result['task_id'] != task_id:
        raise ValueError('Optimizer task execution identity mismatch')
    if result.get('verified'):
        native = result['result']
        source = json.loads(_artifact(store, key, result['candidate']['sha256']))
        baseline = json.loads(_artifact(store, key, result['baseline']['sha256']))
        if (not isinstance(source, str) or sha(source.encode()) != result['source_sha256']
                or native.get('candidate_sha256') != result['source_sha256']
                or native.get('status') != 'verified'
                or native.get('suite_sha256') != baseline['suite_sha256']
                or not baseline['cases']
                or sorted(c['case'] for c in native['cases']) != sorted(c['case'] for c in baseline['cases'])):
            raise ValueError('verified attempt does not match frozen suite and candidate')
