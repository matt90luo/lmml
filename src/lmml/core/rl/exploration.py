"""关闭 NoisyNet 时使用的线性 epsilon-greedy 退火调度器。"""

from __future__ import annotations


class EpsilonSchedule:
    def __init__(self, start: float = 1.0, end: float = 0.05, decay_steps: int = 10000):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        frac = min(1.0, max(0.0, int(step) / self.decay_steps))
        return self.start + frac * (self.end - self.start)
