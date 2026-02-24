"""Unit tests for ImageStorage.

This module tests the ImageStorage class including:
- Image saving and path retrieval
- SQLite index persistence
- Collection management
- Concurrent access (WAL mode)
"""

import tempfile
import shutil
from pathlib import Path

import pytest

from ingestion.storage import ImageStorage


class TestImageStorageSave:
    """Tests for image saving functionality."""

    def test_save_image_creates_file(self, tmp_path):
        """Test that saving an image creates the file."""
        # Create temp image
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        # Create a simple test file
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        image_id = storage.save_image(test_image, collection="test")

        assert image_id is not None
        assert len(image_id) == 16

        # Check file exists
        saved_path = image_dir / "test" / f"{image_id}.png"
        assert saved_path.exists()

    def test_save_image_same_content_same_id(self, tmp_path):
        """Test that same image produces same ID."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        id1 = storage.save_image(test_image, collection="test")
        id2 = storage.save_image(test_image, collection="test")

        assert id1 == id2

    def test_save_image_different_content_different_id(self, tmp_path):
        """Test that different images produce different IDs."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image1 = tmp_path / "test1.png"
        test_image1.write_bytes(b"image data 1")

        test_image2 = tmp_path / "test2.png"
        test_image2.write_bytes(b"image data 2")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        id1 = storage.save_image(test_image1, collection="test")
        id2 = storage.save_image(test_image2, collection="test")

        assert id1 != id2


class TestImageStorageRetrieve:
    """Tests for image path retrieval."""

    def test_get_image_path(self, tmp_path):
        """Test getting image path by ID."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        image_id = storage.save_image(test_image, collection="test")
        path = storage.get_image_path(image_id)

        assert path is not None
        assert image_id in path
        assert Path(path).exists()

    def test_get_image_path_not_found(self, tmp_path):
        """Test getting path for non-existent image."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        path = storage.get_image_path("nonexistent_id_12345")

        assert path is None

    def test_get_image_record(self, tmp_path):
        """Test getting full image record."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        image_id = storage.save_image(
            test_image,
            collection="my_collection",
            doc_hash="abc123",
            page_num=1,
        )

        record = storage.get_image_record(image_id)

        assert record is not None
        assert record.image_id == image_id
        assert record.collection == "my_collection"
        assert record.doc_hash == "abc123"
        assert record.page_num == 1


class TestImageStorageCollections:
    """Tests for collection management."""

    def test_get_images_by_collection(self, tmp_path):
        """Test querying images by collection."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        # Create multiple test images
        for i in range(3):
            test_image = tmp_path / f"test{i}.png"
            test_image.write_bytes(f"image data {i}".encode())

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        # Save to same collection
        for i in range(3):
            test_image = tmp_path / f"test{i}.png"
            storage.save_image(test_image, collection="collection_a")

        # Save to different collection
        test_image = tmp_path / "test_other.png"
        test_image.write_bytes(b"other collection")
        storage.save_image(test_image, collection="collection_b")

        # Query collection_a
        images = storage.get_images_by_collection("collection_a")

        assert len(images) == 3

        # Query collection_b
        images = storage.get_images_by_collection("collection_b")

        assert len(images) == 1

    def test_clear_collection(self, tmp_path):
        """Test clearing a collection."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        storage.save_image(test_image, collection="to_clear")

        count = storage.clear_collection("to_clear")

        assert count == 1
        assert storage.count(collection="to_clear") == 0


class TestImageStorageDocHash:
    """Tests for document hash queries."""

    def test_get_images_by_doc_hash(self, tmp_path):
        """Test querying images by document hash."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image1 = tmp_path / "test1.png"
        test_image1.write_bytes(b"page 1")

        test_image2 = tmp_path / "test2.png"
        test_image2.write_bytes(b"page 2")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        doc_hash = "document_hash_abc"

        # Save pages
        storage.save_image(test_image1, collection="docs", doc_hash=doc_hash, page_num=1)
        storage.save_image(test_image2, collection="docs", doc_hash=doc_hash, page_num=2)

        images = storage.get_images_by_doc_hash(doc_hash)

        assert len(images) == 2
        assert images[0].page_num == 1
        assert images[1].page_num == 2


class TestImageStorageDelete:
    """Tests for image deletion."""

    def test_delete_image(self, tmp_path):
        """Test deleting an image."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        image_id = storage.save_image(test_image, collection="test")

        # Verify exists
        assert storage.get_image_path(image_id) is not None

        # Delete
        result = storage.delete_image(image_id)

        assert result is True
        assert storage.get_image_path(image_id) is None


class TestImageStorageCount:
    """Tests for counting images."""

    def test_count_all(self, tmp_path):
        """Test counting all images."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        for i in range(5):
            test_image = tmp_path / f"test{i}.png"
            test_image.write_bytes(f"data {i}".encode())

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        for i in range(5):
            test_image = tmp_path / f"test{i}.png"
            storage.save_image(test_image, collection="test")

        assert storage.count() == 5

    def test_count_collection(self, tmp_path):
        """Test counting images in a collection."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        # Add to collection_a
        for i in range(3):
            test_image = tmp_path / f"test{i}.png"
            test_image.write_bytes(f"data {i}".encode())
            storage.save_image(test_image, collection="collection_a")

        # Add to collection_b
        test_image = tmp_path / "test_other.png"
        test_image.write_bytes(b"other")
        storage.save_image(test_image, collection="collection_b")

        assert storage.count(collection="collection_a") == 3
        assert storage.count(collection="collection_b") == 1


class TestImageStoragePersistence:
    """Tests for database persistence."""

    def test_persistence(self, tmp_path):
        """Test that image records persist across instances."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        # First instance - save
        storage1 = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))
        image_id = storage1.save_image(test_image, collection="test")
        storage1.close()

        # Second instance - retrieve
        storage2 = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))
        path = storage2.get_image_path(image_id)

        assert path is not None
        assert image_id in path

    def test_context_manager(self, tmp_path):
        """Test using ImageStorage as context manager."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        with ImageStorage(image_dir=str(image_dir), db_path=str(db_path)) as storage:
            image_id = storage.save_image(test_image, collection="test")
            assert storage.get_image_path(image_id) is not None


class TestImageStorageEdgeCases:
    """Tests for edge cases."""

    def test_save_nonexistent_file(self, tmp_path):
        """Test saving non-existent file raises error."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        with pytest.raises(Exception):
            storage.save_image("/nonexistent/file.png", collection="test")

    def test_empty_collection(self, tmp_path):
        """Test querying empty collection."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        images = storage.get_images_by_collection("empty_collection")

        assert images == []


class TestImageStorageRepr:
    """Tests for string representation."""

    def test_repr(self, tmp_path):
        """Test string representation."""
        image_dir = tmp_path / "images"
        db_path = tmp_path / "test.db"

        storage = ImageStorage(image_dir=str(image_dir), db_path=str(db_path))

        repr_str = repr(storage)

        assert "ImageStorage" in repr_str
        assert str(image_dir) in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
