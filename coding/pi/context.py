"""Stable historical JSON encoding; native Pi owns live context and pagination."""
import json


class ContextLimitError(ValueError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
