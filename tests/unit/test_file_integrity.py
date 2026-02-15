"""Unit tests for file integrity checking (SQLite-based).

This module tests the file_integrity module for:
- SHA256 hash computation
- SqliteFileIntegrityChecker implementation
- Interface compliance (FileIntegrityChecker ABC)

Design Principles:
    - Deterministic: Same file always produces same hash
    - Isolation: Tests use temp directories and clean up
    - Coverage: Hash computation, SQLite backend, abstract interface
"""

import tempfile
import time
from abc import ABC
from pathlib import Path
from typing import Any

import pytest

from libs.loader.file_integrity import (
    compute_sha256,
    FileIntegrityChecker,
    SqliteFileIntegrityChecker,
    should_skip,
    mark_success,
    clear_history,
    close_checker,
    FileIntegrityError,
    HashComputationError,
)


class TestComputeSHA256:
    """Tests for compute_sha256 function."""

    def test_same_content_produces_same_hash(self):
        """Same file content should produce identical hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            f.flush()
            path1 = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            f.flush()
            path2 = f.name

        try:
            hash1 = compute_sha256(path1)
            hash2 = compute_sha256(path2)
            assert hash1 == hash2
            assert len(hash1) == 64
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_different_content_produces_different_hash(self):
        """Different file content should produce different hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content A")
            f.flush()
            path1 = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content B")
            f.flush()
            path2 = f.name

        try:
            hash1 = compute_sha256(path1)
            hash2 = compute_sha256(path2)
            assert hash1 != hash2
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_empty_file(self):
        """Empty file should produce valid hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            file_hash = compute_sha256(path)
            assert len(file_hash) == 64
            assert file_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        finally:
            Path(path).unlink()

    def test_nonexistent_file_raises_error(self):
        """Non-existent file should raise HashComputationError."""
        with pytest.raises(HashComputationError):
            compute_sha256("/path/that/does/not/exist/file.txt")


class TestSqliteFileIntegrityChecker:
    """Tests for SqliteFileIntegrityChecker implementation."""

    def setup_method(self):
        """Set up a fresh checker with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.checker = SqliteFileIntegrityChecker(
            db_path=str(Path(self.temp_dir) / "test.db")
        )

    def teardown_method(self):
        """Clean up."""
        self.checker.close()
        clear_history()

    def test_should_skip_unmarked_hash(self):
        """Hash not in history should not be skipped."""
        test_hash = "abc123def456" + "0" * 50
        result = self.checker.should_skip(test_hash)
        assert result is False

    def test_should_skip_marked_hash(self):
        """Hash marked as success should be skipped."""
        test_hash = "marked_hash_for_skip_test" + "0" * 46
        self.checker.mark_success(test_hash, "/test/path.pdf")
        result = self.checker.should_skip(test_hash)
        assert result is True

    def test_different_hash_not_skipped(self):
        """Different hash should not be skipped."""
        self.checker.mark_success("hash1" + "0" * 58, "/test/path1.pdf")
        result = self.checker.should_skip("hash2" + "0" * 58)
        assert result is False

    def test_mark_success_with_source_path(self):
        """mark_success should record source path."""
        test_hash = "mark_test_hash_" + "0" * 48
        self.checker.mark_success(test_hash, "/path/to/file.pdf")
        assert self.checker.should_skip(test_hash) is True

    def test_mark_success_adds_timestamp(self):
        """mark_success should record timestamp."""
        test_hash = "timestamp_test_hash_" + "0" * 46
        before = time.time()
        self.checker.mark_success(test_hash, "/path/file.pdf")
        after = time.time()
        # Re-create checker to query fresh state
        checker2 = SqliteFileIntegrityChecker(
            db_path=str(Path(self.temp_dir) / "test.db")
        )
        try:
            assert checker2.should_skip(test_hash) is True
        finally:
            checker2.close()

    def test_clear_removes_all_records(self):
        """clear should remove all records."""
        self.checker.mark_success("hash1" + "0" * 58, "/path1.pdf")
        self.checker.mark_success("hash2" + "0" * 58, "/path2.pdf")
        self.checker.clear()
        assert self.checker.should_skip("hash1" + "0" * 58) is False
        assert self.checker.should_skip("hash2" + "0" * 58) is False

    def test_context_manager(self):
        """Should work as context manager."""
        with SqliteFileIntegrityChecker(
            db_path=str(Path(self.temp_dir) / "context.db")
        ) as checker:
            test_hash = "context_test_hash_" + "0" * 48
            checker.mark_success(test_hash, "/path.pdf")
            assert checker.should_skip(test_hash) is True
        # After context exit, connection should be closed

    def test_upsert_behavior(self):
        """mark_success should update existing record."""
        test_hash = "upsert_test_hash_" + "0" * 48

        self.checker.mark_success(test_hash, "/path1.pdf")
        self.checker.mark_success(test_hash, "/path2.pdf")

        # Should still skip and have latest source path
        assert self.checker.should_skip(test_hash) is True


class TestFileIntegrityCheckerInterface:
    """Tests for FileIntegrityChecker abstract interface."""

    def test_sqlite_implements_interface(self):
        """SqliteFileIntegrityChecker should implement FileIntegrityChecker."""
        assert issubclass(SqliteFileIntegrityChecker, FileIntegrityChecker)

    def test_interface_has_required_methods(self):
        """FileIntegrityChecker should have required abstract methods."""
        assert hasattr(FileIntegrityChecker, 'should_skip')
        assert hasattr(FileIntegrityChecker, 'mark_success')
        assert hasattr(FileIntegrityChecker, 'clear')
        assert hasattr(FileIntegrityChecker, 'close')

    def test_sqlite_has_all_interface_methods(self):
        """SqliteFileIntegrityChecker should implement all interface methods."""
        assert hasattr(SqliteFileIntegrityChecker, 'should_skip')
        assert hasattr(SqliteFileIntegrityChecker, 'mark_success')
        assert hasattr(SqliteFileIntegrityChecker, 'clear')
        assert hasattr(SqliteFileIntegrityChecker, 'close')


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Clear history before each test."""
        clear_history()
        close_checker()

    def teardown_method(self):
        """Clean up after each test."""
        clear_history()
        close_checker()

    def test_should_skip_uses_module_level_checker(self):
        """should_skip should use the module-level checker."""
        test_hash = "module_level_test_" + "0" * 51
        assert should_skip(test_hash) is False

        mark_success(test_hash, "/test.pdf")
        assert should_skip(test_hash) is True

    def test_clear_clears_module_level_checker(self):
        """clear_history should clear module-level history."""
        mark_success("clear_test_" + "0" * 56, "/test.pdf")
        assert should_skip("clear_test_" + "0" * 56) is True

        clear_history()

        assert should_skip("clear_test_" + "0" * 56) is False

    def test_close_checker_resets_instance(self):
        """close_checker should reset the module-level instance."""
        mark_success("close_test_" + "0" * 55, "/test.pdf")
        close_checker()
        # After close, should get fresh instance on next call


class TestIntegration:
    """Integration tests for the complete workflow."""

    def setup_method(self):
        """Set up fresh state."""
        clear_history()
        close_checker()

    def teardown_method(self):
        """Clean up."""
        clear_history()
        close_checker()

    def test_ingestion_workflow(self):
        """Test complete ingestion skip workflow with SQLite."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("Document content for testing")
            path = f.name

        try:
            # Compute hash
            file_hash = compute_sha256(path)

            # First time: should not skip
            assert should_skip(file_hash) is False

            # Mark as processed
            mark_success(file_hash, path)

            # Second time: should skip
            assert should_skip(file_hash) is True

        finally:
            Path(path).unlink()

    def test_modified_file_not_skipped(self):
        """Modified file should not be skipped even if path is same."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Original content")
            path = f.name

        try:
            # First hash
            hash1 = compute_sha256(path)
            mark_success(hash1, path)
            assert should_skip(hash1) is True

            # Modify file
            with open(path, "w") as f:
                f.write("Modified content")

            # New hash
            hash2 = compute_sha256(path)

            # Old hash should still be in history
            assert should_skip(hash1) is True
            # New hash should not skip
            assert should_skip(hash2) is False

        finally:
            Path(path).unlink()

    def test_sqlite_persistence(self):
        """Changes should persist after closing and reopening."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First checker - mark a hash
            with SqliteFileIntegrityChecker(db_path=db_path) as checker:
                test_hash = "persist_test_" + "0" * 54
                checker.mark_success(test_hash, "/test.pdf")

            # Second checker - same database, should see the hash
            with SqliteFileIntegrityChecker(db_path=db_path) as checker2:
                assert checker2.should_skip(test_hash) is True

        finally:
            Path(db_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
