"""Dependency-free deterministic CPU embeddings for transport tests."""
import hashlib


class Embeddings:
    model = 'deterministic-cpu'
    revision = '1'

    def embed(self, texts):
        return [[value / 255 for value in hashlib.sha256(text.encode()).digest()]
                for text in texts]
