"""File integrity checking via SHA256 hash with SQLite backend.

This module provides file hash computation and skip-check functionality
for incremental ingestion. Files that haven't changed (same hash)
can be skipped to avoid redundant processing.

Design Principles:
    - Deterministic: Same file always produces same hash
    - Fast: Uses hashlib for efficient SHA256 computation
    - Abstract: FileIntegrityChecker interface for extensibility
    - SQLite: Default persistence backend (replaceable)
"""

import hashlib
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# Default cache directory and database
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_DB_FILE = "ingestion_history.db"


class FileIntegrityError(Exception):
    """Base exception for file integrity operations."""

    pass


class HashComputationError(FileIntegrityError):
    """Failed to compute file hash."""

    pass


class DatabaseError(FileIntegrityError):
    """Database operation failed."""

    pass


def compute_sha256(path: str | Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        Hexadecimal string of the SHA256 hash.

    Raises:
        HashComputationError: If file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise HashComputationError(f"File not found: {path}")

    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except IOError as e:
        raise HashComputationError(f"Failed to read file {path}: {e}")


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity checking.

    Implementations provide persistent storage for tracking which files
    have been successfully processed.
    """

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Check if a file should be skipped based on its hash.

        Args:
            file_hash: SHA256 hash of the file.

        Returns:
            True if the file was previously processed successfully.
        """
        pass

    @abstractmethod
    def mark_success(self, file_hash: str, source_path: str | None = None) -> None:
        """Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file.
            source_path: Optional source file path for reference.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all ingestion history."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any database connections."""
        pass


class SqliteFileIntegrityChecker(FileIntegrityChecker):
    """SQLite-based implementation of FileIntegrityChecker.

    Uses a SQLite database to track file hash history for incremental
    ingestion. This provides reliable persistence and query performance.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the SQLite-based integrity checker.

        Args:
            db_path: Path to the SQLite database file.
                     If None, uses default location.
        """
        if db_path is None:
            cache_dir = Path(DEFAULT_CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = cache_dir / DEFAULT_DB_FILE
        else:
            self._db_path = Path(db_path)

        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            # Enable foreign keys and row factory
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_history (
                hash TEXT PRIMARY KEY,
                source_path TEXT,
                status TEXT NOT NULL DEFAULT 'success',
                timestamp REAL NOT NULL,
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON file_history(status)
        """)
        conn.commit()

    def should_skip(self, file_hash: str) -> bool:
        """Check if a file should be skipped.

        Args:
            file_hash: SHA256 hash of the file.

        Returns:
            True if the file was previously processed successfully.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT status FROM file_history WHERE hash = ?",
            (file_hash,)
        )
        row = cursor.fetchone()

        if row is None:
            return False

        return row["status"] == "success"

    def mark_success(self, file_hash: str, source_path: str | None = None) -> None:
        """Mark a file as successfully processed.

        Uses UPSERT to update existing records or insert new ones.

        Args:
            file_hash: SHA256 hash of the file.
            source_path: Optional source file path for reference.
        """
        conn = self._get_connection()
        timestamp = time.time()

        conn.execute("""
            INSERT INTO file_history (hash, source_path, status, timestamp)
            VALUES (?, ?, 'success', ?)
            ON CONFLICT(hash) DO UPDATE SET
                source_path = COALESCE(EXCLUDED.source_path, file_history.source_path),
                status = 'success',
                timestamp = EXCLUDED.timestamp
        """, (file_hash, source_path, timestamp))

        conn.commit()

    def clear(self) -> None:
        """Clear all ingestion history."""
        conn = self._get_connection()
        conn.execute("DELETE FROM file_history")
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SqliteFileIntegrityChecker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures connection is closed."""
        self.close()


# Module-level singleton instance
_checker: SqliteFileIntegrityChecker | None = None


def _get_checker() -> SqliteFileIntegrityChecker:
    """Get or create the module-level checker instance."""
    global _checker
    if _checker is None:
        _checker = SqliteFileIntegrityChecker()
    return _checker


def should_skip(file_hash: str) -> bool:
    """Check if a file should be skipped based on its hash.

    Args:
        file_hash: SHA256 hash of the file.

    Returns:
        True if the file was previously processed successfully.
    """
    return _get_checker().should_skip(file_hash)


def mark_success(file_hash: str, source_path: str | None = None) -> None:
    """Mark a file as successfully processed.

    Args:
        file_hash: SHA256 hash of the file.
        source_path: Optional source file path for reference.
    """
    _get_checker().mark_success(file_hash, source_path)


def clear_history() -> None:
    """Clear all ingestion history.

    This is useful for testing or when a full re-index is needed.
    """
    global _checker
    if _checker is not None:
        _checker.clear()
    _checker = None


def close_checker() -> None:
    """Close the global checker connection."""
    global _checker
    if _checker is not None:
        _checker.close()
        _checker = None
