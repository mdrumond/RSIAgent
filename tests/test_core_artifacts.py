"""Shared CAS preserves legacy artifacts and canonical byte contracts."""
import pytest


def test_lightweight_cas_preserves_legacy_identity_and_canonical_bytes(tmp_path):
    import hashlib
    from dataclasses import dataclass
    from enum import Enum
    from core.artifacts import ContentAddressedArtifacts, canonical_bytes
    from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts as LegacyArtifacts
    from benchmarks.a5kernels.evidence import canonical_bytes as legacy_bytes

    class Label(Enum):
        VALUE = 'example'

    @dataclass
    class Record:
        label: Label
        values: tuple

    value = Record(Label.VALUE, (1., True, None))
    expected = b'{"label":"example","values":[1.0,true,null]}'
    assert canonical_bytes(value) == legacy_bytes(value) == expected
    assert LegacyArtifacts is ContentAddressedArtifacts
    assert ContentAddressedArtifacts(tmp_path).put_json(value) == hashlib.sha256(expected).hexdigest()
    with pytest.raises(ValueError):
        canonical_bytes({'invalid': float('nan')})
    with pytest.raises(TypeError):
        canonical_bytes({1: 'nonstring key'})

