"""Persistent supervised coding process with streamed evidence and group cancellation."""
import multiprocessing
import atexit
import os
import signal
import threading
import time

from .control import WorkerStopped


def _serve(agent, pipe, inherited_parent, parent_pid, deadline):
    inherited_parent.close()
    os.setsid()
    # Pi joins this supervisor's group so Stop covers descendants as well.
    agent.own_process_group = False
    live = callable(getattr(deadline, 'endpoint', None))
    if live:
        from benchmarks.ascend.actor_time import detach_actor_budget
        detach_actor_budget()
    agent.deadline = None if live else deadline
    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(.2)
        # The worker disappeared, including SIGKILL: stop the whole owned group.
        os.killpg(os.getpid(), signal.SIGKILL)
    threading.Thread(target=watch_parent, daemon=True).start()
    pipe.send(('ready',))
    try:
        while True:
            if not live and time.monotonic() >= deadline:
                break
            if not pipe.poll(.2):
                continue
            command = pipe.recv()
            if command is None:
                break
            stage, context = command[:2]
            if live:
                agent.deadline = command[2]
                session = getattr(agent, 'session', None)
                if session is not None:
                    session.deadline = command[2]
                    session.gateway.deadline = command[2]
            try:
                value = agent.complete(stage, context, lambda kind, data: pipe.send(('record', kind, data)))
                pipe.send(('result', value))
            except Exception as exc:
                pipe.send(('error', type(exc).__name__))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        close = getattr(agent, 'close', None)
        if close is not None:
            close()
        pipe.close()


class BoundedAgent:
    def __init__(self, agent, *, deadline, stopped, clock=time.monotonic):
        self.agent, self.identity = agent, agent.identity
        self.deadline, self.stopped, self.clock = deadline, stopped, clock
        self.process = self.pipe = None
        self.ready = False
        atexit.register(self.close)

    def _check(self):
        if self.stopped() or self.clock() >= self.deadline:
            raise WorkerStopped('provider stopped or wall-time budget exhausted')

    def _receive(self):
        while True:
            self._check()
            if self.pipe.poll(.1):
                try:
                    return self.pipe.recv()
                except EOFError:
                    raise RuntimeError('provider process exited without a result') from None
            if not self.process.is_alive():
                raise RuntimeError('provider process exited without a result')

    def complete(self, stage, context, record):
        try:
            self._check()
            if self.process is None:
                process_context = multiprocessing.get_context('fork')
                self.pipe, child = process_context.Pipe()
                self.process = process_context.Process(target=_serve, args=(
                    self.agent, child, self.pipe, os.getpid(), self.deadline))
                self.process.start()
                child.close()
                if self._receive() != ('ready',):
                    raise RuntimeError('invalid provider supervisor handshake')
                self.ready = True
            endpoint = getattr(self.deadline, 'endpoint', None)
            self.pipe.send((stage, context, endpoint()) if callable(endpoint) else (stage, context))
            while True:
                message = self._receive()
                if message[0] == 'record':
                    record(message[1], message[2])
                elif message[0] == 'result':
                    return message[1]
                else:
                    raise RuntimeError(f'provider {message[1]}')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.is_alive():
                try:
                    self.pipe.send(None)
                except (BrokenPipeError, OSError):
                    pass
                self.process.join(timeout=.2)
            # Kill the group even if the supervisor exited; a crashed parent
            # must not leave Pi/provider readers alive.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if self.process.is_alive():
                    self.process.terminate()
            self.process.join(timeout=2)
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=2)
        finally:
            self.pipe.close()
            self.process = self.pipe = None
            self.ready = False
