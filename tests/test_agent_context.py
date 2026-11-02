"""Historical profile metadata; conversation/compaction behavior uses native SDK tests."""
import pytest
from benchmarks.ascend.agent_context import ContextBudget, request_bytes

def test_request_size_counts_unicode_and_wire_metadata():
    assert request_bytes({"messages": [{"content": "é"}]}) > len("é")
    with pytest.raises(ValueError):
        ContextBudget(1024, 2000, 3000)

