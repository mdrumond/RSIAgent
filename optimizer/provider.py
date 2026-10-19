"""Bound provider requests in a child process while streaming retained records."""
import multiprocessing
import time

from .transport import WorkerStopped


def _complete(agent, stage, context, pipe):
    try:
        value = agent.complete(stage, context, lambda kind, data: pipe.send(('record', kind, data)))
        pipe.send(('result', value))
    except Exception as exc:
        pipe.send(('error', type(exc).__name__, str(exc)))
    finally:
        pipe.close()


class BoundedAgent:
    def __init__(self, agent, *, deadline, stopped, clock=time.monotonic):
        self.agent, self.identity = agent, agent.identity
        self.deadline, self.stopped, self.clock = deadline, stopped, clock

    def complete(self, stage, context, record):
        # Fork preserves the configured provider object without serializing secrets.
        process_context = multiprocessing.get_context('fork')
        parent, child = process_context.Pipe(duplex=False)
        process = process_context.Process(target=_complete, args=(self.agent, stage, context, child))
        process.start()
        child.close()
        try:
            while True:
                if self.stopped() or self.clock() >= self.deadline:
                    raise WorkerStopped('provider stopped or wall-time budget exhausted')
                if parent.poll(.2):
                    try:
                        message = parent.recv()
                    except EOFError:
                        raise RuntimeError('provider process exited without a result')
                    if message[0] == 'record':
                        record(message[1], message[2])
                    elif message[0] == 'result':
                        return message[1]
                    else:
                        raise RuntimeError(f'provider {message[1]}: {message[2]}')
                elif not process.is_alive():
                    raise RuntimeError('provider process exited without a result')
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            parent.close()
