"""Helpers for loading built-in modules and external plugins."""

from __future__ import annotations

import importlib
import pkgutil


def import_submodules(package_name: str) -> None:
    """Import all non-private modules below a package."""
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    prefix = f"{package_name}."
    for module_info in pkgutil.iter_modules(package_path, prefix):
        leaf = module_info.name.rsplit(".", maxsplit=1)[-1]
        if leaf.startswith("_") or leaf in {"base"}:
            continue
        importlib.import_module(module_info.name)
