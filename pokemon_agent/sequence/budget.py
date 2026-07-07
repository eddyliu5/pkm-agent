from __future__ import annotations

import time

from .types import SequenceSearchConfig


class SearchBudget:
    def __init__(self, config: SequenceSearchConfig):
        self.config = config
        self.started_at = time.monotonic()
        self.nodes = 0
        self.node_budget_hit = False
        self.time_budget_hit = False

    def can_expand(self) -> bool:
        if self.nodes >= self.config.max_nodes:
            self.node_budget_hit = True
            return False
        if self.timed_out():
            self.time_budget_hit = True
            return False
        return True

    def record_node(self) -> None:
        self.nodes += 1
        if self.nodes >= self.config.max_nodes:
            self.node_budget_hit = True

    def timed_out(self) -> bool:
        hit = (time.monotonic() - self.started_at) * 1000.0 >= float(self.config.time_budget_ms)
        self.time_budget_hit = self.time_budget_hit or hit
        return hit
