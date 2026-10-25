"""可配置的棋盘 Gymnasium 环境。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP, DOWN, LEFT, RIGHT, STAY = range(5)
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), STAY: (0, 0)}
_CHANNELS = ("agent", "target", "trap", "wall")


def _cells(cells: Sequence[Sequence[int]] | None) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in (cells or ()))


class GridWorldEnv(gym.Env):
    """
    棋盘环境。
    - 状态: 整数 id = row * n_cols + col (obs_mode="discrete")
            或 (row, col) 的 Box (obs_mode="coords")
            或 (4, H, W) 的多通道网格 agent/target/trap/wall (obs_mode="image")
    - 动作: 0上 1下 2左 3右 4不动
    - 终止: 到 target -> terminated=True, reward=target_reward
            到 trap   -> terminated=True, reward=trap_reward
            步数超限   -> truncated=True
    - 撞墙: 原地不动, reward = step_reward + wall_penalty
            (边界与内部 walls 格子都算撞墙)
    """

    metadata: ClassVar[dict] = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        start: Sequence[int] | str = (0, 0),
        targets: Sequence[Sequence[int]] | None = None,
        traps: Sequence[Sequence[int]] | None = None,
        walls: Sequence[Sequence[int]] = (),
        step_reward: float = -0.01,
        wall_penalty: float = -0.1,
        target_reward: float = 1.0,
        trap_reward: float = -1.0,
        max_steps: int = 100,
        obs_mode: str = "discrete",
        render_mode: str | None = None,
    ):
        super().__init__()
        if obs_mode not in ("discrete", "coords", "image"):
            raise ValueError(f"obs_mode must be discrete/coords/image, got {obs_mode!r}")
        self.n_rows, self.n_cols = int(n_rows), int(n_cols)
        if targets is None:
            targets = ((self.n_rows - 1, self.n_cols - 1),)
        if traps is None:
            mid = (self.n_rows // 2, self.n_cols // 2)
            traps = (mid,) if mid not in _cells(targets) and mid != (0, 0) else ()
        self.targets, self.traps, self.walls = _cells(targets), _cells(traps), _cells(walls)
        for name, cells in (("targets", self.targets), ("traps", self.traps), ("walls", self.walls)):
            for r, c in cells:
                if not self._inside(r, c):
                    raise ValueError(f"{name} cell {(r, c)} is outside the {n_rows}x{n_cols} grid")
        overlap = (set(self.targets) & set(self.traps)) | (set(self.walls) & (set(self.targets) | set(self.traps)))
        if overlap:
            raise ValueError(f"target/trap/wall cells overlap: {sorted(overlap)}")
        self.start = "random" if start == "random" else _cells([start])[0]
        if self.start != "random" and (self.start in self.walls or not self._inside(*self.start)):
            raise ValueError(f"invalid start cell {self.start}")
        self.step_reward, self.wall_penalty = float(step_reward), float(wall_penalty)
        self.target_reward, self.trap_reward = float(target_reward), float(trap_reward)
        self.max_steps = int(max_steps)
        self.obs_mode = obs_mode
        self.render_mode = render_mode

        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 5
        self.action_space = spaces.Discrete(self.n_actions)
        if obs_mode == "discrete":
            self.observation_space = spaces.Discrete(self.n_states)
        elif obs_mode == "coords":
            high = np.array([self.n_rows - 1, self.n_cols - 1], dtype=np.float32)
            self.observation_space = spaces.Box(0.0, high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                0.0, 1.0, (len(_CHANNELS), self.n_rows, self.n_cols), dtype=np.float32
            )
        self._layers = self._static_layers()
        self._free_starts = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.walls and (r, c) not in self.targets and (r, c) not in self.traps
        ]
        self._pos = (0, 0)
        self._steps = 0

    # ------------------------------------------------------------------ helpers
    def _inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def to_state(self, rc: Sequence[int]) -> int:
        return int(rc[0]) * self.n_cols + int(rc[1])

    def to_rc(self, s: int) -> tuple[int, int]:
        return int(s) // self.n_cols, int(s) % self.n_cols

    def _static_layers(self) -> np.ndarray:
        layers = np.zeros((len(_CHANNELS), self.n_rows, self.n_cols), dtype=np.float32)
        for r, c in self.targets:
            layers[1, r, c] = 1.0
        for r, c in self.traps:
            layers[2, r, c] = 1.0
        for r, c in self.walls:
            layers[3, r, c] = 1.0
        return layers

    def _move(self, rc: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        """返回 (新位置, 是否撞墙)。"""
        dr, dc = _DELTAS[int(action)]
        nr, nc = rc[0] + dr, rc[1] + dc
        if not self._inside(nr, nc) or (nr, nc) in self.walls:
            return rc, True
        return (nr, nc), False

    def _obs(self):
        r, c = self._pos
        if self.obs_mode == "discrete":
            return self.to_state(self._pos)
        if self.obs_mode == "coords":
            return np.array([r, c], dtype=np.float32)
        img = self._layers.copy()
        img[0, r, c] = 1.0
        return img

    def transition_model(self) -> np.ndarray:
        """next_state[s, a] (终止格 target/trap 视为吸收态)。"""
        nxt = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        absorbing = set(self.targets) | set(self.traps)
        for s in range(self.n_states):
            rc = self.to_rc(s)
            for a in range(self.n_actions):
                nxt[s, a] = s if rc in absorbing or rc in self.walls else self.to_state(self._move(rc, a)[0])
        return nxt

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.start == "random":
            self._pos = self._free_starts[int(self.np_random.integers(len(self._free_starts)))]
        else:
            self._pos = self.start
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action {action!r}")
        self._pos, hit_wall = self._move(self._pos, int(action))
        self._steps += 1
        reward = self.step_reward + (self.wall_penalty if hit_wall else 0.0)
        terminated = truncated = False
        info = {"is_success": False}
        if self._pos in self.targets:
            terminated, reward, info["is_success"] = True, self.target_reward, True
        elif self._pos in self.traps:
            terminated, reward = True, self.trap_reward
        elif self._steps >= self.max_steps:
            truncated = True
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        rows = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                rc = (r, c)
                row += (
                    "A" if rc == self._pos else "T" if rc in self.targets else "X" if rc in self.traps
                    else "#" if rc in self.walls else "."
                )
            rows.append(row)
        return "\n".join(rows)
