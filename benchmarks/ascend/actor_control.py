"""Durable user instructions and safe-boundary actor continuation.

The inbox belongs to an execution workspace, never to a provider session. A
receipt means an instruction is retained in the next request's context, not
that a provider completed or obeyed it.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import time

from optimizer.jobs import atomic, canonical


class ActorStopped(RuntimeError):
    pass


class ActorControl:
    def __init__(self, workspace):
        self.root = Path(workspace) if workspace is not None else None
        self.path = self.root / 'actor-control.json' if self.root is not None else None
        self.transient = dict(schema='actor-control-v1', messages=[], checkpoint=None,
                              stop_requested=False)

    @contextmanager
    def transaction(self):
        if self.root is None:
            yield self.transient
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.actor-control.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_bytes()) if self.path.exists() else {
                'schema': 'actor-control-v1', 'messages': [], 'checkpoint': None,
                'stop_requested': False,
            }
            yield state
            atomic(self.path, canonical(state))

    def send(self, key, text):
        if not isinstance(key, str) or not key.strip():
            raise ValueError('feedback requires an idempotency key')
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 16384:
            raise ValueError('feedback must contain 1..16384 UTF-8 bytes')
        with self.transaction() as state:
            for row in state['messages']:
                if row['key'] == key:
                    if row['text'] != text:
                        raise ValueError('feedback key reused with different text')
                    return dict(row)
            row = dict(id=len(state['messages']) + 1, key=key, text=text,
                       created=time.time(), consumed=None)
            state['messages'].append(row)
            return dict(row)

    def status(self):
        with self.transaction() as state:
            return json.loads(canonical(state))

    def pending(self):
        state = self.status()
        return state['stop_requested'] or any(row['consumed'] is None for row in state['messages'])

    def stop(self, requested=True):
        with self.transaction() as state:
            state['stop_requested'] = bool(requested)

    def save(self, checkpoint):
        with self.transaction() as state:
            state['checkpoint'] = checkpoint

    def boundary(self, checkpoint, feedback):
        """Atomically retain context and receipt before issuing a new request."""
        with self.transaction() as state:
            if state['stop_requested']:
                state['checkpoint'] = checkpoint
                # Commit the checkpoint before notifying the actor.
                stopped = True
            else:
                stopped = False
                for row in state['messages']:
                    if row['consumed'] is None:
                        feedback.append(dict(role='user', content=row['text'],
                                             feedback_id=row['id']))
                        row['consumed'] = time.time()
                state['checkpoint'] = checkpoint
        if stopped:
            raise ActorStopped('user requested a stop at the next safe boundary')


class ActorConversation:
    """Checkpoint model context without restoring host verification authority."""
    def __init__(self, workspace, instruction, working, action_executor=None):
        self.control = ActorControl(workspace)
        self.identity = hashlib.sha256(instruction.encode()).hexdigest()
        self.working, self.actions = working, action_executor
        self.turn = 0
        self.feedback = []
        checkpoint = self.control.status()['checkpoint']
        if checkpoint is not None:
            if checkpoint['instruction_sha256'] != self.identity:
                raise ValueError('actor instruction changed; start a fresh linked attempt')
            self.turn = checkpoint['turn']
            self.feedback = checkpoint.get('feedback', [])
            for field in ('fixed', 'recent', 'work_log', 'host_state', 'tail'):
                setattr(working, field, checkpoint['working'][field])
            source = checkpoint.get('source')
            restore = getattr(action_executor, 'restore_source', None)
            if source is not None and restore is not None:
                restore(source)
                working.host_state = {}
                working.append(dict(role='user', content=(
                    'The actor resumed from retained source and conversation. '
                    'Host verification and measurements must be repeated before submission.')))
        self._pin()

    def _pin(self):
        # WorkingContext never compacts pinned messages. IDs are metadata only.
        self.working.pinned = [{key: row[key] for key in ('role', 'content')}
                               for row in self.feedback]

    def snapshot(self):
        return dict(instruction_sha256=self.identity, turn=self.turn,
            feedback=self.feedback, source=getattr(self.actions, 'source', None),
            working={field: getattr(self.working, field) for field in
                     ('fixed', 'recent', 'work_log', 'host_state', 'tail')})

    def boundary(self):
        checkpoint = self.snapshot()
        self.control.boundary(checkpoint, self.feedback)
        self._pin()
        return self.feedback

    def save(self, turn):
        self.turn = turn
        self.control.save(self.snapshot())


def grant_elapsed(path, key, additional_seconds):
    """Extend a settled elapsed allocation once, retaining previous usage."""
    import math
    if type(additional_seconds) not in (int, float) or not math.isfinite(additional_seconds) or additional_seconds <= 0:
        raise ValueError('additional elapsed seconds must be finite and positive')
    if not isinstance(key, str) or not key.strip():
        raise ValueError('allocation requires an idempotency key')
    path = Path(path)
    with path.with_suffix(path.suffix + '.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_bytes())
        grants = state.setdefault('grants', {})
        if key in grants:
            if grants[key] != additional_seconds:
                raise ValueError('allocation key reused with different seconds')
            return state['max_seconds']
        if state.get('active_since') is not None:
            raise ValueError('settle the running elapsed interval before extending it')
        grants[key] = additional_seconds
        state['max_seconds'] += additional_seconds
        atomic(path, canonical(state))
        return state['max_seconds']
