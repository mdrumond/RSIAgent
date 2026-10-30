"""Execution control boundaries shared by the loop and production adapters."""


class ObservationUnavailable(RuntimeError):
    """Retain pending work and observe its original operation identity."""


class WorkerStopped(RuntimeError):
    """Stop or elapsed-time budget boundary; no new work may be submitted."""


class ObservedBudgetStop(WorkerStopped):
    """Elapsed stop with an exact, retained terminal transport observation."""
    def __init__(self, message, receipt_path):
        from pathlib import Path
        from hashlib import sha256
        import json
        path = Path(receipt_path).resolve(strict=True)
        raw = path.read_bytes()
        state = json.loads(raw)
        if state.get('state') != 'cancelled' or not state.get('native_handle'):
            raise ValueError('budget stop requires observed cancellation of the original native handle')
        super().__init__(message)
        self.terminal_receipt = dict(path=str(path), sha256=sha256(raw).hexdigest(),
                                     handle=state['native_handle'], state='cancelled')

    def validate(self, root):
        from pathlib import Path
        from hashlib import sha256
        import json
        path = Path(self.terminal_receipt['path']).resolve(strict=True)
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError('observed budget receipt is outside the admitted execution root')
        raw = path.read_bytes()
        state = json.loads(raw)
        if (sha256(raw).hexdigest() != self.terminal_receipt['sha256']
                or state.get('state') != 'cancelled'
                or state.get('native_handle') != self.terminal_receipt['handle']):
            raise ValueError('observed budget cancellation receipt changed')
        return self.terminal_receipt
