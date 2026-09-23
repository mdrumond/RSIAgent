from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchmarks.a5kernels.embeddings import EmbeddingLoadError, PinnedBGEEmbeddings
from benchmarks.a5kernels.kdb_cli import DEFAULT_BACKEND, _backend as load_backend
from benchmarks.a5kernels.knowledge import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
)


class RecordingTokenizer:
    def __init__(self):
        self.batches = []

    def __call__(self, texts, **kwargs):
        self.batches.append((texts, kwargs))
        token = MagicMock()
        token.to.return_value = token
        return {"input_ids": token, "attention_mask": token}


class FakeTorch:
    def __init__(self):
        self.deterministic = None

    def use_deterministic_algorithms(self, enabled):
        self.deterministic = enabled

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeEncoder:
    def __init__(self, *, revision=DEFAULT_EMBEDDING_REVISION, dimension=384):
        self.config = SimpleNamespace(_commit_hash=revision, hidden_size=dimension)
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self

    def __call__(self, *, input_ids, attention_mask):
        batch_size = len(self.tokenizer.batches[-1][0])
        states = MagicMock()
        pooled = MagicMock()
        states.__mul__.return_value.sum.return_value.__truediv__.return_value = pooled
        rows = [[float(index + 1), 2.0] * 192 for index in range(batch_size)]
        pooled.detach.return_value.cpu.return_value.tolist.return_value = rows
        return SimpleNamespace(last_hidden_state=states)


def _fake_backend(tokenizer, encoder, **kwargs):
    encoder.tokenizer = tokenizer
    calls = []

    def load_tokenizer(model, **load_kwargs):
        calls.append(("tokenizer", model, load_kwargs))
        return tokenizer

    def load_model(model, **load_kwargs):
        calls.append(("model", model, load_kwargs))
        return encoder

    return PinnedBGEEmbeddings(
        tokenizer_loader=load_tokenizer,
        model_loader=load_model,
        torch_module=FakeTorch(),
        **kwargs,
    ), calls


def test_loader_pins_identity_revision_cache_and_offline_mode(tmp_path):
    tokenizer = RecordingTokenizer()
    encoder = FakeEncoder()
    backend, calls = _fake_backend(tokenizer, encoder, cache_dir=tmp_path / "../cache")

    backend.embed(["kernel"])

    assert [call[0] for call in calls] == ["tokenizer", "model"]
    for _, model, kwargs in calls:
        assert model == DEFAULT_EMBEDDING_MODEL
        assert kwargs == {
            "revision": DEFAULT_EMBEDDING_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "cache_dir": str((tmp_path / "../cache").resolve()),
        }
    assert encoder.device == "cpu"
    assert encoder.evaluating


def test_embed_batches_mean_pools_and_returns_unit_vectors():
    tokenizer = RecordingTokenizer()
    backend, _ = _fake_backend(tokenizer, FakeEncoder(), batch_size=2)

    vectors = backend.embed(["one", "two", "three", "four", "five"])

    assert [len(batch) for batch, _ in tokenizer.batches] == [2, 2, 1]
    assert len(vectors) == 5
    assert all(len(vector) == 384 for vector in vectors)
    assert all(math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0) for vector in vectors)
    assert tokenizer.batches[0][1] == {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (FakeEncoder(revision=None), "provenance is unverified"),
        (FakeEncoder(revision="main"), "provenance is unverified"),
        (FakeEncoder(dimension=768), "unexpected embedding dimension"),
    ],
)
def test_loader_fails_closed_on_unverified_revision_or_dimension(encoder, message):
    backend, _ = _fake_backend(RecordingTokenizer(), encoder)

    with pytest.raises(EmbeddingLoadError, match=message):
        backend.embed(["kernel"])


def test_offline_cache_miss_has_context_and_preserves_cause():
    def missing(_model, **kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("not cached")

    backend = PinnedBGEEmbeddings(
        tokenizer_loader=missing,
        model_loader=missing,
        torch_module=FakeTorch(),
    )

    with pytest.raises(EmbeddingLoadError, match="from local cache") as raised:
        backend.embed(["kernel"])
    assert isinstance(raised.value.__cause__, OSError)


def test_empty_input_does_not_load_model():
    def unexpected(*_args, **_kwargs):
        raise AssertionError("empty input must not load")

    backend = PinnedBGEEmbeddings(
        tokenizer_loader=unexpected,
        model_loader=unexpected,
        torch_module=FakeTorch(),
    )

    assert backend.embed([]) == []


def test_cli_default_resolves_to_offline_pinned_backend():
    backend = load_backend(DEFAULT_BACKEND)

    assert isinstance(backend, PinnedBGEEmbeddings)
    assert backend.local_files_only is True


def test_real_tensor_pooling_ignores_padding_and_is_repeatable():
    torch = pytest.importorskip("torch")

    class TensorTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 99]] * len(texts)),
                "attention_mask": torch.tensor([[1, 1, 0]] * len(texts)),
            }

    class TensorEncoder(FakeEncoder):
        def __call__(self, *, input_ids, attention_mask):
            assert not torch.is_grad_enabled()
            assert input_ids.device.type == "cpu"
            states = torch.zeros((len(input_ids), 3, 384))
            states[:, 0, 0] = 6
            states[:, 1, 1] = 8
            states[:, 2, 2] = 1000  # A padded token must not affect the vector.
            return SimpleNamespace(last_hidden_state=states)

    encoder = TensorEncoder()
    backend = PinnedBGEEmbeddings(
        batch_size=2,
        tokenizer_loader=lambda *_args, **_kwargs: TensorTokenizer(),
        model_loader=lambda *_args, **_kwargs: encoder,
        torch_module=torch,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        first = backend.embed(["one", "two", "three"])
        assert first == backend.embed(["one", "two", "three"])
        assert torch.are_deterministic_algorithms_enabled()
        assert first == [[0.6, 0.8] + [0.0] * 382] * 3
    finally:
        torch.use_deterministic_algorithms(previous)
