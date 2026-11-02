"""Historical exception import aliases; execution uses the native SDK boundary."""
from .native import NativeError as PiError, ProviderFailure as PiProviderError
from .native import DeadlineExceeded as PiDeadlineError, Cancelled as PiCancelled


class PiCompactionError(PiProviderError):
    """Historical classification, readable without launching an old RPC engine."""
