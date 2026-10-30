"""Controller-clock phase scopes for existing synchronous domain transports."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

CURRENT = ContextVar('canonical_domain_activity', default=None)


@contextmanager
def activity_phase(category):
    scope = CURRENT.get()
    if scope is None or scope['category'] == category:
        yield
        return
    flow, previous = scope['flow'], scope['category']
    flow.accounting.end(scope['token'])
    scope['token'] = flow.accounting.start(category)
    scope['category'] = category
    try:
        yield
    finally:
        flow.accounting.end(scope['token'])
        scope['token'] = flow.accounting.start(previous)
        scope['category'] = previous


@contextmanager
def excluded_phase(category='transfer'):
    with activity_phase(category):
        yield


def transfer_phase(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with excluded_phase():
            return function(*args, **kwargs)
    return wrapped


def canonical_allowance():
    scope = CURRENT.get()
    if scope is None or scope['category'] in scope['flow'].accounting.EXCLUDED:
        return None
    values = [value for value in scope['flow'].remaining().values() if value is not None]
    return max(0., min(values)) if values else None


def bounded_deadline(wall_deadline):
    import time
    remaining = canonical_allowance()
    return wall_deadline if remaining is None else min(wall_deadline, time.monotonic() + remaining)
