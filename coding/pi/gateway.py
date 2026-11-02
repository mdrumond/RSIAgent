"""Historical context-admission error type; no executable provider gateway."""
from .context import ContextLimitError


class RequestContextLimit(ContextLimitError):
    """Historical classification retained for old evidence readers."""
