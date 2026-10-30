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
        if category in flow.accounting.EXCLUDED and scope.get('budget_update') is not None:
            scope['budget_update']()


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
