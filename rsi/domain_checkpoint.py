"""Lossless host-owned native values in the existing CAS; never pickle/code replay."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import importlib
import math
from pathlib import Path


def encode(value):
    if isinstance(value, Enum):
        return dict(native_type='enum', module=type(value).__module__, name=type(value).__qualname__, value=encode(value.value))
    if is_dataclass(value) and not isinstance(value, type):
        return dict(native_type='dataclass', module=type(value).__module__, name=type(value).__qualname__,
                    fields={field.name: encode(getattr(value, field.name)) for field in fields(value)},
                    provenance_present=getattr(value, '_provenance_present', None))
    if isinstance(value, Mapping):
        return dict(native_type='mapping', items=[[encode(k), encode(v)] for k,v in value.items()])
    if isinstance(value, tuple):
        return dict(native_type='tuple', items=[encode(v) for v in value])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, Path):
        return dict(native_type='path', value=str(value))
    if isinstance(value, bytes):
        return dict(native_type='bytes', value=value.hex())
    if isinstance(value, float) and not math.isfinite(value):
        return dict(native_type='float', value=repr(value))
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError('unsupported native checkpoint value: '+type(value).__qualname__)


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value['native_type']
    if kind == 'mapping':
        return {decode(k):decode(v) for k,v in value['items']}
    if kind == 'tuple':
        return tuple(decode(v) for v in value['items'])
    if kind == 'path':
        return Path(value['value'])
    if kind == 'bytes':
        return bytes.fromhex(value['value'])
    if kind == 'float' and value['value'] in ('nan','inf','-inf'):
        return float(value['value'])
    if kind not in ('dataclass','enum') or not value['module'].startswith(('benchmarks.', 'rsi.')) or '<locals>' in value['name']:
        raise ValueError('unregistered native checkpoint type')
    cls = importlib.import_module(value['module'])
    for part in value['name'].split('.'):
        cls = getattr(cls, part)
    if kind == 'enum' and isinstance(cls, type) and issubclass(cls, Enum):
        return cls(decode(value['value']))
    if kind == 'dataclass' and is_dataclass(cls):
        declared = {field.name:field for field in fields(cls)}
        if set(value['fields']) != set(declared):
            raise ValueError('native checkpoint dataclass fields differ')
        arguments = {k:decode(v) for k,v in value['fields'].items() if declared[k].init}
        if value.get('provenance_present') is not None:
            arguments['provenance_present'] = value['provenance_present']
        result = cls(**arguments)
        if any(encode(getattr(result,k)) != v for k,v in value['fields'].items()):
            raise ValueError('native checkpoint derived fields differ')
        return result
    raise ValueError('native checkpoint type does not match schema')


def has_pending_result(value):
    """Inspect current response envelopes, not historical diagnostic status rows."""
    if is_dataclass(value) and not isinstance(value, type):
        current = {field.name:getattr(value,field.name) for field in fields(value)
                   if field.name in ('state','status','observation')}
        return has_pending_result(current)
    if isinstance(value, Mapping):
        if any(value.get(key) in ('pending','queued','running','unknown','unresolved','dispatching','ambiguous')
               for key in ('state','status') if isinstance(value.get(key), str)):
            return True
        for key in ('host','result','observation'):
            child = value.get(key)
            if key == 'observation' and isinstance(child,str):
                import json
                try:
                    child = json.loads(child)
                except ValueError:
                    continue
            if has_pending_result(child):
                return True
        return False
    if isinstance(value, (tuple,list)):
        return any(has_pending_result(v) for v in value)
    return False
