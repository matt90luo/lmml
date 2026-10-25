"""向量化 SumTree，支持单点与批量更新、批量采样及 O(N) 重建。"""

from __future__ import annotations

import numpy as np


class SumTree:
    """用一维 float64 数组维护完全二叉树（避免大量求和累积误差）。

    叶子位于 ``tree[capacity - 1 : 2 * capacity - 1]``，与数据槽位 ``0..capacity-1`` 一一对应。
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.size = 0

    @property
    def leaf_offset(self) -> int:
        return self.capacity - 1

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def min_priority(self) -> float:
        if self.size == 0:
            return 0.0
        leaves = self.tree[self.leaf_offset : self.leaf_offset + self.size]
        pos = leaves[leaves > 0]
        return float(pos.min()) if pos.size > 0 else 1e-8

    def add(self, priority: float) -> int:
        """在当前 write_ptr 位置写入优先级，返回数据槽位索引 [0, capacity)。"""
        data_idx = self.write_ptr
        tree_idx = data_idx + self.leaf_offset
        self.update(tree_idx, float(priority))
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        return data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        tree_idx = int(tree_idx)
        change = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def update_batch(self, tree_indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(np.asarray(tree_indices, dtype=np.int64), np.asarray(priorities, dtype=np.float64)):
            self.update(int(idx), float(prio))

    def set_leaves_bulk(self, data_indices: np.ndarray, priorities: np.ndarray) -> None:
        """批量设置叶子优先级并 O(N) 自底向上重建内部节点。"""
        self.tree[self.leaf_offset + data_indices] = priorities
        for i in range(self.leaf_offset - 1, -1, -1):
            left = 2 * i + 1
            self.tree[i] = self.tree[left] + self.tree[left + 1]

    def get_leaf(self, value: float) -> tuple[int, float, int]:
        """返回 (tree_idx, priority, data_idx)。"""
        parent = 0
        n = len(self.tree)
        val = float(value)
        while True:
            left = 2 * parent + 1
            if left >= n:
                leaf = parent
                break
            if val <= self.tree[left]:
                parent = left
            else:
                val -= self.tree[left]
                parent = left + 1
        data_idx = leaf - self.leaf_offset
        if data_idx >= self.size:
            data_idx = max(0, self.size - 1)
            leaf = data_idx + self.leaf_offset
        return leaf, float(self.tree[leaf]), data_idx

    def get_leaves(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """向量化批量检索叶子，返回 (tree_indices, priorities, data_indices)。"""
        vals = np.asarray(values, dtype=np.float64).copy()
        idx = np.zeros(len(vals), dtype=np.int64)
        n = len(self.tree)
        while True:
            left = 2 * idx + 1
            has_child = left < n
            if not np.any(has_child):
                break
            active = np.flatnonzero(has_child)
            l_idx = left[active]
            l_val = self.tree[l_idx]
            go_right = vals[active] > l_val
            vals[active] = np.where(go_right, vals[active] - l_val, vals[active])
            idx[active] = np.where(go_right, l_idx + 1, l_idx)
        data_idx = idx - self.leaf_offset
        if self.size > 0:
            over = data_idx >= self.size
            if np.any(over):
                data_idx[over] = self.size - 1
                idx[over] = data_idx[over] + self.leaf_offset
        return idx, self.tree[idx].astype(np.float32), data_idx
