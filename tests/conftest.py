"""
Pytest configuration and shared fixtures for the Modular RAG MCP Server test suite.

This module provides common fixtures used across unit, integration, and e2e tests.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add the project root and src directories to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

for path in [str(SRC_ROOT), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the path to the config directory."""
    return project_root / "config"


@pytest.fixture(scope="session")
def data_path(project_root: Path) -> Path:
    """Return the path to the data directory."""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def fixtures_path(project_root: Path) -> Path:
    """Return the path to the tests fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_documents_path(fixtures_path: Path) -> Path:
    """Return the path to the sample documents fixtures directory."""
    return fixtures_path / "sample_documents"
