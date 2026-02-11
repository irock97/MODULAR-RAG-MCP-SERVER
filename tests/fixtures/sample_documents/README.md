# Sample Documents for Testing

This directory contains sample documents used for testing the Modular RAG MCP Server.

## Usage

Place test documents here for use in unit and integration tests. Supported formats:
- PDF files (.pdf)
- Markdown files (.md)
- Text files (.txt)

## Guidelines

1. Use small, focused documents for unit tests
2. Include documents with images for testing image processing
3. Use documents with tables for testing table extraction
4. Keep file sizes small (< 1MB) for fast test execution

## Example Documents

When adding test documents, consider:
- `sample_article.md` - A simple markdown article for basic text processing
- `sample_with_images.pdf` - PDF with embedded images for image captioning tests
- `sample_with_tables.pdf` - PDF with tables for table extraction tests
