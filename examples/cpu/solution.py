"""A known-good CPU fixture; task generators can seed a naive sum instead."""
import math


def stable_sum(values):
    return math.fsum(values)
