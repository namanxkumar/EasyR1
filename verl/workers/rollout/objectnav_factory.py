"""ObjectNav dataset wrapper for multi-turn GRPO rollouts.

Provides dataset iteration for SimulatorPool-based env creation.
Actual ObjectNavEnvironment and ObjectNavEnvAdapter creation happens
inside SimulatorPool.acquire_env().
"""

from __future__ import annotations

import numpy as np

from .multiturn_env import EnvFactory


class ObjectNavEnvFactory(EnvFactory):
    """Provides dataset items for SimulatorPool-based env creation.

    Only manages dataset iteration. Actual ObjectNavEnvironment and
    ObjectNavEnvAdapter creation happens inside SimulatorPool.acquire_env().
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._dataset_len = len(dataset)
        self._indices = list(range(self._dataset_len))
        self._item_idx = 0

    def __len__(self) -> int:
        return self._dataset_len

    def get_next_item(self) -> dict:
        """Return the next dataset item (cycling through the dataset)."""
        if self._item_idx >= self._dataset_len:
            np.random.shuffle(self._indices)
            self._item_idx = 0
        data = self.dataset[self._indices[self._item_idx]]
        self._item_idx += 1
        return data
