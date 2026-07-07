from __future__ import annotations

__all__ = ["ObservationBuilder"]


def __getattr__(name: str):
    if name == "ObservationBuilder":
        from .builder import ObservationBuilder

        return ObservationBuilder
    raise AttributeError(name)
