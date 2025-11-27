"""
PDF Parser Adapter

Wrapper around DocumentParser to match expected import path.
"""

from graphplag.core.document_parser import DocumentParser
from graphplag.utils.file_parser import FileParser


class PDFParser:
    """
    PDF Parser adapter that wraps DocumentParser.
    Provides a simple interface for parsing PDF and other document types.
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize PDF parser.
        
        Args:
            language: Language code for document parsing
        """
        self.language = language
        self.parser = DocumentParser(language=language)
        self.file_parser = FileParser()
    
    def parse(self, file_path: str) -> str:
        """
        Parse a document file and extract text.
        
        Args:
            file_path: Path to document file (PDF, DOCX, TXT, etc.)
            
        Returns:
            Extracted text content
        """
        try:
            # Use FileParser to extract text from file
            text = self.file_parser.parse_file(file_path)
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse file {file_path}: {str(e)}")
    
    def parse_text(self, text: str) -> str:
        """
        Parse raw text (identity function for compatibility).
        
        Args:
            text: Raw text
            
        Returns:
            Same text
        """
        return text
    
    def __repr__(self) -> str:
        return f"PDFParser(language='{self.language}')"
