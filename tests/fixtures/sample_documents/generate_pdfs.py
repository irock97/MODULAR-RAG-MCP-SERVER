"""Sample PDF generator for testing.

This module provides utility functions to generate sample PDF documents
for testing the PdfLoader.

Usage:
    from generate_pdfs import generate_sample_pdf, generate_pdf_with_images

Requirements:
    pip install reportlab
"""

from pathlib import Path
from typing import Optional


def generate_sample_pdf(
    output_path: str | Path,
    title: str = "Sample Document",
    content: str | None = None,
) -> Path:
    """Generate a simple text-based PDF for testing.

    Args:
        output_path: Path where the PDF will be saved.
        title: Title of the document.
        content: Optional custom content. If None, generates default content.

    Returns:
        Path to the generated PDF file.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        raise ImportError(
            "reportlab is required. Install with: pip install reportlab"
        )

    if content is None:
        content = f"""
{title}

This is a sample PDF document for testing the PdfLoader.

Section 1: Introduction
This document contains various sections to test the PDF loader functionality.
It includes multiple paragraphs to test text extraction.

Section 2: Technical Details
The PDF loader should be able to extract this text and convert it to markdown format.
This is important for the ingestion pipeline.

Section 3: Testing
Various test cases will verify that:
1. Text extraction works correctly
2. Metadata is properly extracted
3. Document structure is preserved
        """.strip()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, title)

    # Add content
    c.setFont("Helvetica", 12)
    y = height - 120

    for line in content.split('\n'):
        if y < 72:  # New page if at bottom
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 72
        c.drawString(72, y, line)
        y -= 18

    c.save()
    return path


def generate_pdf_with_images(
    output_path: str | Path,
    title: str = "Document with Images",
    num_images: int = 2,
) -> tuple[Path, list[str]]:
    """Generate a PDF with embedded images for testing image extraction.

    Args:
        output_path: Path where the PDF will be saved.
        title: Title of the document.
        num_images: Number of images to embed in the PDF.

    Returns:
        Tuple of (Path to PDF, list of image filenames created).
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise ImportError(
            "reportlab is required. Install with: pip install reportlab"
        )

    from PIL import Image
    import io

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample images
    image_files = []
    for i in range(num_images):
        img_path = path.parent / f"sample_image_{i}.png"
        # Create a simple colored image using PIL
        img = Image.new('RGB', (200, 100), color=(73 + i * 50, 109 + i * 30, 137))
        img.save(str(img_path))
        image_files.append(str(img_path))

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, title)

    # Add text and images
    c.setFont("Helvetica", 12)
    y = height - 120

    c.drawString(72, y, "This document contains embedded images for testing.")
    y -= 30

    for i, img_file in enumerate(image_files):
        if y < 200:  # New page if not enough space
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 72

        c.drawString(72, y, f"Image {i + 1}:")
        y -= 20

        try:
            img = ImageReader(img_file)
            c.drawImage(img, 72, y - 50, width=100, height=50)
        except Exception as e:
            c.drawString(72, y, f"[Image {i + 1} could not be embedded: {e}]")
        y -= 80

    c.save()
    return path, image_files


if __name__ == "__main__":
    # Generate sample PDFs
    fixtures_dir = Path(__file__).parent

    # Generate simple PDF
    pdf_path = fixtures_dir / "sample_document.pdf"
    generate_sample_pdf(pdf_path)
    print(f"Generated: {pdf_path}")

    # Generate PDF with images
    pdf_with_images = fixtures_dir / "document_with_images.pdf"
    path, images = generate_pdf_with_images(pdf_with_images)
    print(f"Generated: {path}")
    print(f"Generated images: {images}")
