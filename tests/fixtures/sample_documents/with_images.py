"""Sample PDF with images for testing.

This module contains information about the sample PDF with images
used for testing PdfLoader's image extraction capabilities.

The PDF should be generated using generate_pdfs.py and contains:
- Title and headings
- Multiple paragraphs of text
- Embedded images with captions
- Various formatting elements

Expected behavior when loaded:
1. Text content extracted as Markdown
2. Images extracted and saved to data/images/{doc_hash}/
3. Image placeholders inserted in markdown: [IMAGE: image_id]
4. Image metadata stored in Document.metadata.images
"""

# Sample document metadata template
SAMPLE_DOC_METADATA = {
    "title": "Document with Images",
    "description": "A sample PDF containing both text and images for testing",
    "expected_images": 2,
    "sections": [
        {
            "name": "Introduction",
            "content": "This document contains various sections to test the PDF loader.",
        },
        {
            "name": "Images Section",
            "content": "The following images demonstrate image extraction capabilities.",
        },
    ],
}

# Expected image placeholders that should appear in extracted text
EXPECTED_IMAGE_PLACEHOLDERS = [
    "[IMAGE: image_0]",
    "[IMAGE: image_1]",
]

# Usage example
EXAMPLE_USAGE = '''
from libs.loader.pdf_loader import PdfLoader

# Load the PDF
loader = PdfLoader(extract_images=True)
doc = loader.load("tests/fixtures/sample_documents/document_with_images.pdf")

# Access extracted content
print(doc.text)           # Markdown content with image placeholders
print(doc.metadata)       # Metadata including image info
'''
