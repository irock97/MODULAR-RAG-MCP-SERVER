"""ImageStorage - Image file storage with SQLite index.

This module provides the ImageStorage class for storing images
and tracking their paths in a SQLite database.

Design Principles:
    - Deterministic: Same image content always produces same image_id
    - Persistent: Image paths tracked in SQLite for fast lookup
    - Concurrent-safe: Uses WAL mode for concurrent database access

Example:
    >>> from ingestion.storage import ImageStorage
    >>> storage = ImageStorage()
    >>> image_id = storage.save_image("/path/to/image.png", "my_collection")
    >>> path = storage.get_image_path(image_id)
"""

import hashlib
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from observability.logger import get_logger

logger = get_logger(__name__)

# Default directories
DEFAULT_IMAGE_DIR = "data/images"
DEFAULT_DB_FILE = "data/db/image_index.db"


@dataclass
class ImageRecord:
    """Image record from database."""

    image_id: str
    file_path: str
    collection: str | None
    doc_hash: str | None
    page_num: int | None
    created_at: str


class ImageStorageError(Exception):
    """Base exception for image storage operations."""

    pass


class ImageNotFoundError(ImageStorageError):
    """Raised when image is not found."""

    pass


class ImageStorage:
    """Image file storage with SQLite index.

    This class handles storing images to the filesystem and maintaining
    a SQLite index for fast path lookups.

    Attributes:
        image_dir: Base directory for storing images.
        db_path: Path to SQLite database file.

    Example:
        >>> from ingestion.storage import ImageStorage
        >>> storage = ImageStorage()
        >>> image_id = storage.save_image("/path/to/image.png", "docs")
        >>> path = storage.get_image_path(image_id)
    """

    def __init__(
        self,
        image_dir: str = DEFAULT_IMAGE_DIR,
        db_path: str | Path | None = None,
    ) -> None:
        """Initialize ImageStorage.

        Args:
            image_dir: Base directory for storing images.
            db_path: Path to SQLite database file.
                     If None, uses default location.
        """
        self._image_dir = Path(image_dir)
        self._db_path = Path(db_path) if db_path else Path(DEFAULT_DB_FILE)

        self._conn: sqlite3.Connection | None = None

        # Ensure directories exist
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

        logger.info(
            f"ImageStorage initialized: image_dir={self._image_dir}, db_path={self._db_path}"
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent access
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection
                ON image_index(collection)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash
                ON image_index(doc_hash)
            """)
            conn.commit()
        except Exception:
            conn.close()
            self._conn = None
            raise

    def _generate_image_id(
        self,
        source_path: str,
        page_num: int | None = None,
    ) -> str:
        """Generate deterministic image ID from source path and page.

        Args:
            source_path: Path to the source document.
            page_num: Page number (for multi-page documents).

        Returns:
            Deterministic image ID string.
        """
        id_source = f"{source_path}"
        if page_num is not None:
            id_source += f":{page_num}"

        return hashlib.sha256(id_source.encode("utf-8")).hexdigest()[:16]

    def _get_collection_dir(self, collection: str) -> Path:
        """Get the directory for a collection."""
        return self._image_dir / collection

    def save_image(
        self,
        source_path: str | Path,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
    ) -> str:
        """Save an image to storage.

        This method copies the image to the collection directory and
        records the mapping in the database.

        Args:
            source_path: Path to the source image file.
            collection: Collection name for organizing images.
            doc_hash: Optional hash of the source document.
            page_num: Optional page number for multi-page documents.

        Returns:
            The generated image_id.

        Raises:
            ImageStorageError: If the source file doesn't exist or can't be copied.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise ImageStorageError(f"Source image not found: {source_path}")

        # Generate image_id
        image_id = self._generate_image_id(str(source_path), page_num)

        # Get collection directory
        collection_dir = self._get_collection_dir(collection)
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        extension = source_path.suffix.lower()
        if not extension:
            extension = ".png"  # Default extension

        # Destination path
        dest_filename = f"{image_id}{extension}"
        dest_path = collection_dir / dest_filename

        # Copy file if it doesn't exist
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.debug(f"Copied image to {dest_path}")
        else:
            logger.debug(f"Image already exists at {dest_path}")

        # Record in database
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    collection = EXCLUDED.collection,
                    doc_hash = EXCLUDED.doc_hash,
                    page_num = EXCLUDED.page_num
            """, (image_id, str(dest_path), collection, doc_hash, page_num))
            conn.commit()
        except Exception:
            conn.close()
            self._conn = None
            raise

        logger.info(f"Saved image {image_id} to {dest_path}")

        return image_id

    def get_image_path(self, image_id: str) -> str | None:
        """Get the file path for an image_id.

        Args:
            image_id: The image identifier.

        Returns:
            The file path if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return row["file_path"]
        except Exception:
            conn.close()
            self._conn = None
            raise

    def get_image_record(self, image_id: str) -> ImageRecord | None:
        """Get the full image record.

        Args:
            image_id: The image identifier.

        Returns:
            ImageRecord if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return ImageRecord(
                image_id=row["image_id"],
                file_path=row["file_path"],
                collection=row["collection"],
                doc_hash=row["doc_hash"],
                page_num=row["page_num"],
                created_at=row["created_at"],
            )
        except Exception:
            conn.close()
            self._conn = None
            raise

    def get_images_by_collection(self, collection: str) -> list[ImageRecord]:
        """Get all images in a collection.

        Args:
            collection: Collection name.

        Returns:
            List of ImageRecords in the collection.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM image_index WHERE collection = ? ORDER BY created_at DESC",
                (collection,)
            )
            rows = cursor.fetchall()

            return [
                ImageRecord(
                    image_id=row["image_id"],
                    file_path=row["file_path"],
                    collection=row["collection"],
                    doc_hash=row["doc_hash"],
                    page_num=row["page_num"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        except Exception:
            conn.close()
            self._conn = None
            raise

    def get_images_by_doc_hash(self, doc_hash: str) -> list[ImageRecord]:
        """Get all images for a document hash.

        Args:
            doc_hash: Document hash.

        Returns:
            List of ImageRecords for the document.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM image_index WHERE doc_hash = ? ORDER BY page_num ASC",
                (doc_hash,)
            )
            rows = cursor.fetchall()

            return [
                ImageRecord(
                    image_id=row["image_id"],
                    file_path=row["file_path"],
                    collection=row["collection"],
                    doc_hash=row["doc_hash"],
                    page_num=row["page_num"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        except Exception:
            conn.close()
            self._conn = None
            raise

    def delete_image(self, image_id: str) -> bool:
        """Delete an image and its database record.

        Args:
            image_id: The image identifier.

        Returns:
            True if deleted, False if not found.
        """
        # Get the file path first
        file_path = self.get_image_path(image_id)
        if file_path is None:
            return False

        # Delete file
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted image file: {path}")

        # Delete database record
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM image_index WHERE image_id = ?", (image_id,))
            conn.commit()
        except Exception:
            conn.close()
            self._conn = None
            raise

        logger.info(f"Deleted image {image_id}")
        return True

    def clear_collection(self, collection: str) -> int:
        """Clear all images in a collection.

        Args:
            collection: Collection name.

        Returns:
            Number of images deleted.
        """
        # Get all images in collection
        images = self.get_images_by_collection(collection)

        # Delete files
        for image in images:
            path = Path(image.file_path)
            if path.exists():
                path.unlink()

        # Delete database records
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE collection = ?",
                (collection,)
            )
            conn.commit()

            count = cursor.rowcount
            logger.info(f"Cleared collection {collection}: {count} images deleted")
            return count
        except Exception:
            conn.close()
            self._conn = None
            raise

    def count(self, collection: str | None = None) -> int:
        """Count images in storage.

        Args:
            collection: Optional collection filter.

        Returns:
            Number of images.
        """
        conn = self._get_connection()
        try:
            if collection is None:
                cursor = conn.execute("SELECT COUNT(*) as count FROM image_index")
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM image_index WHERE collection = ?",
                    (collection,)
                )
            row = cursor.fetchone()
            return row["count"] if row else 0
        except Exception:
            conn.close()
            self._conn = None
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ImageStorage":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures connection is closed."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ImageStorage(image_dir={self._image_dir}, db_path={self._db_path})"
        )
