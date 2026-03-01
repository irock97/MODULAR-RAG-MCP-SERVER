"""Integration tests for QueryPipeline.

These tests verify the query script works correctly by running it as a subprocess.
"""

import sys
from pathlib import Path

import pytest


class TestQueryScriptExecution:
    """Integration tests that run the actual query.py script as subprocess.

    These tests allow modifying commands in the test file and get the same
    effect as running in the terminal. Uses real config (no mocks).

    Usage:
        Modify the command parameters below to test different scenarios:
        - CHANGE_QUERY: Query text to search for
        - CHANGE_COLLECTION: Collection name
        - CHANGE_TOP_K: Number of results to return
        - CHANGE_CONFIG: Config file path
        - CHANGE_NO_RERANK: Skip reranking

    Example:
        # Test basic query
        python -m pytest tests/integration/test_query_pipeline.py::TestQueryScriptExecution::test_query_script_basic -v -s

        # Test with verbose output
        python -m pytest tests/integration/test_query_pipeline.py::TestQueryScriptExecution::test_query_script_verbose -v -s --capture=no
    """

    # ===== MODIFY THESE PARAMETERS TO TEST DIFFERENT SCENARIOS =====
    # These can be changed to test different query scenarios
    CHANGE_QUERY = "what does the retrieval system implements?"
    CHANGE_COLLECTION = "test_collection"
    CHANGE_TOP_K = 4
    CHANGE_CONFIG = "config/settings.yaml"
    CHANGE_NO_RERANK = False  # Set to True to skip reranking
    # =============================================================

    @pytest.fixture
    def project_root(self) -> Path:
        """Return project root."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def config_path(self, project_root: Path) -> Path:
        """Return path to config file."""
        config = project_root / self.CHANGE_CONFIG
        assert config.exists(), f"Config file not found: {config}"
        return config

    def test_query_script_basic(self, project_root, config_path):
        """Test running query.py as subprocess for basic search.

        This test executes the exact same command as would be run in terminal:
            python scripts/query.py --query <query> --collection <collection> --top-k <k> --config <config>

        Args:
            Modify class parameters to test different scenarios.
        """
        import subprocess

        # Build the command - same as terminal
        cmd = [
            sys.executable,  # Use current Python interpreter
            str(project_root / "scripts" / "query.py"),
            "--query", self.CHANGE_QUERY,
            "--collection", self.CHANGE_COLLECTION,
            "--top-k", str(self.CHANGE_TOP_K),
            "--config", str(config_path),
        ]

        if self.CHANGE_NO_RERANK:
            cmd.append("--no-rerank")

        print(f"\n[TEST] Executing command:")
        print(f"  $ {' '.join(cmd)}")

        # Run the command - same effect as terminal
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        # Print output (same as terminal)
        print("\n[STDOUT]")
        print(result.stdout)
        if result.stderr:
            print("\n[STDERR]")
            print(result.stderr)

        print(f"\n[EXIT CODE] {result.returncode}")

        # Verify results
        # Exit code 0 = success
        assert result.returncode == 0, (
            f"Query script failed with exit code {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

        # Verify query appears in output
        assert self.CHANGE_QUERY in result.stdout or "Query:" in result.stdout, (
            "Expected query not found in output"
        )

        print("\n[PASS] Query script executed successfully!")

    def test_query_script_verbose_mode(self, project_root, config_path):
        """Test verbose mode - shows intermediate results."""
        import subprocess

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "query.py"),
            "--query", self.CHANGE_QUERY,
            "--collection", self.CHANGE_COLLECTION,
            "--top-k", str(self.CHANGE_TOP_K),
            "--config", str(config_path),
            "--verbose",
        ]

        if self.CHANGE_NO_RERANK:
            cmd.append("--no-rerank")

        print(f"\n[TEST] Executing verbose command:")
        print(f"  $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        print("\n[STDOUT]")
        print(result.stdout)

        assert result.returncode == 0, f"Verbose query failed: {result.stderr}"

        # Verbose should show intermediate results
        assert "DENSE RETRIEVAL" in result.stdout or "FUSION" in result.stdout or "RESULTS" in result.stdout

        print("\n[PASS] Verbose mode executed successfully!")

    def test_query_script_skip_rerank(self, project_root, config_path):
        """Test query with --no-rerank flag."""
        import subprocess

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "query.py"),
            "--query", self.CHANGE_QUERY,
            "--collection", self.CHANGE_COLLECTION,
            "--top-k", str(self.CHANGE_TOP_K),
            "--config", str(config_path),
            "--no-rerank",
        ]

        print(f"\n[TEST] Executing query without reranking:")
        print(f"  $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        print("\n[STDOUT]")
        print(result.stdout)

        assert result.returncode == 0, f"Query without rerank failed: {result.stderr}"

        # Should show reranking disabled
        assert "disabled" in result.stdout.lower() or "Rerank:" in result.stdout

        print("\n[PASS] Skip rerank test executed successfully!")

    def test_query_script_with_custom_top_k(self, project_root, config_path):
        """Test query with custom top-k value."""
        import subprocess

        custom_top_k = 5
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "query.py"),
            "--query", self.CHANGE_QUERY,
            "--collection", self.CHANGE_COLLECTION,
            "--top-k", str(custom_top_k),
            "--config", str(config_path),
            "--no-rerank",
        ]

        print(f"\n[TEST] Executing query with custom top-k:")
        print(f"  $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        print("\n[STDOUT]")
        print(result.stdout)

        assert result.returncode == 0, f"Custom top-k query failed: {result.stderr}"

        # Verify top-k appears in output
        assert str(custom_top_k) in result.stdout or "Top-K:" in result.stdout

        print("\n[PASS] Custom top-k test executed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
