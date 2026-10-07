"""Tiny plugin registry used for reactions, spectrometers and radiation models."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._items)}") from None

    def names(self) -> list[str]:
        return sorted(self._items)
