from __future__ import annotations

import itertools

from .action_prior import PriorAction


class CandidateDiversifier:
    """Preserve first-action and motif coverage while applying budget caps."""

    BUCKET_ORDER = ("attack", "attach", "evolve", "search", "draw", "gust", "retreat", "ability", "end", "other")

    @staticmethod
    def selection_counts(min_count: int, max_count: int) -> list[int]:
        if max_count <= 0:
            return [0] if min_count == 0 else []
        if min_count == max_count:
            return [min_count]
        values = {min_count, max_count}
        if min_count <= 1 <= max_count:
            values.add(1)
        return sorted(value for value in values if min_count <= value <= max_count)

    def keep_action_pool(self, scored_actions: list[PriorAction], *, option_pool_size: int) -> list[PriorAction]:
        if not scored_actions:
            return []
        capacity = max(1, int(option_pool_size))
        if len(scored_actions) <= capacity:
            return list(scored_actions)

        kept: list[PriorAction] = []
        used: set[int] = set()
        by_bucket: dict[str, list[PriorAction]] = {bucket: [] for bucket in self.BUCKET_ORDER}
        for item in scored_actions:
            by_bucket.setdefault(item.bucket, []).append(item)

        for bucket in self.BUCKET_ORDER:
            if len(kept) >= capacity:
                break
            items = by_bucket.get(bucket) or []
            if items:
                kept.append(items[0])
                used.add(items[0].action.index)

        for item in scored_actions:
            if len(kept) >= capacity:
                break
            if item.action.index in used:
                continue
            kept.append(item)
            used.add(item.action.index)
        return kept

    def combinations(self, indices: list[int], counts: list[int], *, max_combinations: int) -> list[tuple[int, ...]]:
        combos: list[tuple[int, ...]] = []
        if 0 in counts:
            combos.append(())
        for count in counts:
            if count <= 0:
                continue
            combos.extend(itertools.combinations(indices, count))
            if len(combos) >= max_combinations:
                break
        return combos[: max(1, int(max_combinations))]
