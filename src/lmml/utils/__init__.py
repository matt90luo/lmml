"""Small runtime utilities shared across lmml experiments."""

from typing import Any

__all__ = ["PeakMemoryTracker", "estimate_peak_vram_gb", "find_optimal_mini_batch"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from lmml.utils import vram_estimator

        return getattr(vram_estimator, name)
    raise AttributeError(name)
