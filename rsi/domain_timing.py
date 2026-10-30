"""Controller-clock phase scopes for existing synchronous domain transports."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

CURRENT = ContextVar('canonical_domain_activity', default=None)


@contextmanager
def excluded_phase(category='transfer'):
    scope = CURRENT.get()
    if scope is None or scope['excluded']:
        yield
        return
    flow = scope['flow']
    flow.accounting.end(scope['token'])
    scope['token'] = flow.accounting.start(category)
    scope['excluded'] = True
    try:
        yield
    finally:
        flow.accounting.end(scope['token'])
        scope['token'] = flow.accounting.start(scope['category'])
        scope['excluded'] = False


def transfer_phase(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with excluded_phase():
            return function(*args, **kwargs)
    return wrapped
