from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


class _PeriodicProgress(Iterator[T]):
    def __init__(self, iterable: Iterable[T], *, desc: str, unit: str, interval: int = 10_000):
        self._iter = iter(iterable)
        self._desc = desc
        self._unit = unit
        self._interval = max(1, int(interval))
        self._count = 0
        self._start = time.monotonic()

    def __iter__(self) -> "_PeriodicProgress[T]":
        return self

    def __next__(self) -> T:
        try:
            item = next(self._iter)
        except StopIteration:
            self._write(final=True)
            raise
        self._count += 1
        if self._count % self._interval == 0:
            self._write(final=False)
        return item

    def _write(self, *, final: bool) -> None:
        elapsed = max(1e-6, time.monotonic() - self._start)
        rate = self._count / elapsed
        suffix = " done" if final else ""
        print(f"{self._desc}: {self._count} {self._unit} ({rate:.1f} {self._unit}/s){suffix}", file=sys.stderr)


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str = "it",
    total: int | None = None,
    enabled: bool = False,
) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, unit=unit, total=total, dynamic_ncols=True)
    except Exception:
        return _PeriodicProgress(iterable, desc=desc, unit=unit)
