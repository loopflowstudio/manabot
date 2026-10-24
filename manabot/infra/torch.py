"""Scoped deterministic CPU settings for bounded fitting and evaluation."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def deterministic_cpu() -> Iterator[None]:
    """Make bounded CPU evidence replay-stable without leaking global settings."""

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
