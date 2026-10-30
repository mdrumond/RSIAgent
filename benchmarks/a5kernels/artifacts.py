"""Compatibility alias for the shared lightweight artifact store."""

from core.artifacts import ContentAddressedArtifacts, canonical_bytes

__all__ = ["ContentAddressedArtifacts", "canonical_bytes"]
