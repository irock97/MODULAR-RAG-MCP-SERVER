# Modular RAG MCP Server
#
# This package contains the core modules for the Modular RAG MCP Server.
# See DEV_SPEC.md for architecture details.

from importlib import import_module
from types import ModuleType
from typing import Iterator


def iter_submodules(package: str, exclude: tuple[str, ...] = ()) -> Iterator[tuple[str, ModuleType]]:
    """Iterate over all submodules of a package.

    Args:
        package: The package name to iterate over.
        exclude: Tuple of module names to exclude.

    Yields:
        Tuples of (module_name, module) for each submodule.
    """
    for importer, modname, ispkg in importlib.util.iter_modules(
        importlib.import_module(package).__path__, package + "."
    ):
        if modname not in exclude:
            mod = importlib.import_module(modname)
            yield modname, mod
            if ispkg:
                yield from iter_submodules(modname, exclude)


import importlib

__all__ = [
    "mcp_server",
    "core",
    "ingestion",
    "libs",
    "observability",
]
