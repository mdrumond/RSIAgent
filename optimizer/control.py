"""Execution control boundaries shared by the loop and production adapters."""


class ObservationUnavailable(RuntimeError):
    """Retain pending work and observe its original operation identity."""


class WorkerStopped(RuntimeError):
    """Stop or elapsed-time budget boundary; no new work may be submitted."""
