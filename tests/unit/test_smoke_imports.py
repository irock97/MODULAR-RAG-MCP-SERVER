"""
Smoke tests for verifying core package imports.

These tests ensure that all top-level packages can be imported correctly
before running more comprehensive test suites.
"""

import importlib
import importlib.resources
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
    try:
        pkg_files = importlib.resources.files(package)
    except (ImportError, TypeError):
        return

    if not pkg_files.is_dir():
        return

    for item in pkg_files.iterdir():
        if item.name.startswith("_"):
            continue
        if item.is_dir():
            subpkg_name = f"{package}.{item.name}"
            if subpkg_name not in exclude:
                try:
                    subpkg = import_module(subpkg_name)
                    yield subpkg_name, subpkg
                    yield from iter_submodules(subpkg_name, exclude)
                except ImportError:
                    pass


class TestCoreImports:
    """Test that all core packages can be imported."""

    def test_import_mcp_server(self) -> None:
        """Verify mcp_server package imports successfully."""
        import mcp_server  # noqa: F401
        assert mcp_server is not None

    def test_import_core(self) -> None:
        """Verify core package imports successfully."""
        import core  # noqa: F401
        assert core is not None

    def test_import_ingestion(self) -> None:
        """Verify ingestion package imports successfully."""
        import ingestion  # noqa: F401
        assert ingestion is not None

    def test_import_libs(self) -> None:
        """Verify libs package imports successfully."""
        import libs  # noqa: F401
        assert libs is not None

    def test_import_observability(self) -> None:
        """Verify observability package imports successfully."""
        import observability  # noqa: F401
        assert observability is not None


class TestSubmoduleDiscovery:
    """Test that submodules can be discovered and imported."""

    def test_mcp_server_submodules(self) -> None:
        """Verify mcp_server package can enumerate its submodules."""
        submodules = list(iter_submodules("mcp_server"))
        # At minimum, should have server and tools submodules
        module_names = [name for name, _ in submodules]
        assert any("server" in name for name in module_names)

    def test_core_submodules(self) -> None:
        """Verify core package can enumerate its submodules."""
        submodules = list(iter_submodules("core"))
        module_names = [name for name, _ in submodules]
        # Should have query_engine, response, trace
        assert any("query" in name for name in module_names)
        assert any("response" in name for name in module_names)

    def test_ingestion_submodules(self) -> None:
        """Verify ingestion package can enumerate its submodules."""
        submodules = list(iter_submodules("ingestion"))
        module_names = [name for name, _ in submodules]
        # Should have embedding, storage, transform
        assert any("embedding" in name for name in module_names)
        assert any("storage" in name for name in module_names)
        assert any("transform" in name for name in module_names)

    def test_libs_submodules(self) -> None:
        """Verify libs package can enumerate its submodules."""
        submodules = list(iter_submodules("libs"))
        module_names = [name for name, _ in submodules]
        # Should have llm, embedding, vector_store, splitter, reranker, evaluator, loader
        assert any("llm" in name for name in module_names)
        assert any("embedding" in name for name in module_names)
        assert any("vector" in name for name in module_names)
        assert any("splitter" in name for name in module_names)

    def test_observability_submodules(self) -> None:
        """Verify observability package can enumerate its submodules."""
        submodules = list(iter_submodules("observability"))
        module_names = [name for name, _ in submodules]
        # Should have dashboard, evaluation
        assert any("dashboard" in name for name in module_names)
        assert any("evaluation" in name for name in module_names)
