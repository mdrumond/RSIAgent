"""Portable evidence paths; live validation uses the production optimizer."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".codex/local/pi-investigation"
DEFAULT_KNOWLEDGE = Path(__file__).parent / "knowledge" / "ascendc.md"


def compose_request(request, knowledge):
    return request + "\nStarting knowledge:\n" + knowledge if knowledge else request
