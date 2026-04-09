"""Plugin-friendly registry implementation."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import threading
from typing import Any, Generic, TypeVar

from guard_eval_harness.plugins.discovery import import_submodules


T = TypeVar("T")
Placeholder = str


class Registry(Generic[T]):
    """Thread-safe registry with optional lazy loading."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T | Placeholder] = {}
        self._lock = threading.RLock()

    def register(self, *aliases: str, target: T | Placeholder | None = None):
        """Register a value directly or use the method as a decorator."""

        def store(alias: str, value: T | Placeholder) -> None:
            current = self._entries.get(alias)
            if current is not None and current != value:
                if (
                    isinstance(current, str)
                    and not isinstance(value, str)
                    and current == f"{value.__module__}:{value.__name__}"
                ):
                    self._entries[alias] = value
                    return
                raise ValueError(
                    f"{self._name!r} alias '{alias}' already registered"
                )
            self._entries[alias] = value

        def decorator(obj: T) -> T:
            names = aliases or (getattr(obj, "__name__", repr(obj)),)
            with self._lock:
                for alias in names:
                    store(alias, obj)
            return obj

        if target is not None:
            if len(aliases) != 1:
                raise ValueError("direct registration requires exactly one alias")
            with self._lock:
                store(aliases[0], target)
            return lambda value: value

        return decorator

    def discover_entry_points(self, group: str) -> None:
        """Register entry points lazily under a registry."""
        for entry_point in metadata.entry_points(group=group):
            self.register(entry_point.name, target=f"{entry_point.module}:{entry_point.attr}")

    def get(self, alias: str) -> T:
        """Fetch a value by alias, materializing lazy placeholders."""
        with self._lock:
            if alias not in self._entries:
                available = ", ".join(sorted(self._entries))
                raise KeyError(
                    f"Unknown {self._name} '{alias}'. Available: {available}"
                )
            value = self._entries[alias]
            if isinstance(value, str):
                module_name, _, attr_name = value.partition(":")
                if not attr_name:
                    raise ValueError(
                        f"invalid placeholder for {self._name} '{alias}'"
                    )
                materialized = getattr(importlib.import_module(module_name), attr_name)
                self._entries[alias] = materialized
                return materialized
            return value

    def items(self) -> list[tuple[str, T]]:
        """Return sorted registry items."""
        return [(alias, self.get(alias)) for alias in sorted(self._entries)]

    def keys(self) -> list[str]:
        """Return sorted registered aliases."""
        return sorted(self._entries)

    def __contains__(self, alias: str) -> bool:
        return alias in self._entries

    def __len__(self) -> int:
        return len(self._entries)


dataset_registry: Registry[Any] = Registry("dataset")
model_registry: Registry[Any] = Registry("model")
_builtins_loaded = False


def ensure_builtin_registrations() -> None:
    """Import built-in dataset and model modules once."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    import_submodules("guard_eval_harness.datasets")
    import_submodules("guard_eval_harness.models")
    dataset_registry.discover_entry_points("guard_eval_harness.datasets")
    model_registry.discover_entry_points("guard_eval_harness.models")
    _builtins_loaded = True
