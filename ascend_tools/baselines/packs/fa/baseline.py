"""Unoptimized PyTorch SDPA; frozen small FA contract, no historical candidate."""
import torch

def baseline(q, k, v, causal=False):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

def oracle(q, k, v, causal=False):
    q, k, v = q.double(), k.double(), v.double()
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** .5
    if causal:
        scores = scores.masked_fill(torch.ones(scores.shape[-2:], dtype=torch.bool).triu(1), -float('inf'))
    return torch.softmax(scores, -1) @ v
