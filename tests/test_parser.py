"""
Unit tests for DocumentParser module.
"""

import pytest
from graphplag.core.document_parser import DocumentParser
from graphplag.core.models import Document, Sentence, LanguageCode


@pytest.fixture
def parser():
    """Create a DocumentParser instance for testing."""
    return DocumentParser(language='en')


def test_parser_initialization(parser):
    """Test parser initializes correctly."""
    assert parser.language == 'en'
    assert parser.nlp is not None


def test_parse_simple_document(parser):
    """Test parsing a simple document."""
    text = "This is a test. This is another sentence."
    document = parser.parse_document(text, doc_id="test_doc")
    
    assert isinstance(document, Document)
    assert document.doc_id == "test_doc"
    assert len(document.sentences) == 2
    assert document.text == text


def test_extract_sentences(parser):
    """Test sentence extraction."""
    text = "First sentence. Second sentence! Third sentence?"
    document = parser.parse_document(text)
    
    assert len(document.sentences) == 3
    
    for i, sent in enumerate(document.sentences):
        assert isinstance(sent, Sentence)
        assert sent.index == i
        assert len(sent.tokens) > 0
        assert len(sent.lemmas) > 0
        assert len(sent.pos_tags) > 0


def test_dependency_extraction(parser):
    """Test dependency extraction."""
    text = "The cat sat on the mat."
    document = parser.parse_document(text)
    
    assert len(document.sentences) == 1
    sentence = document.sentences[0]
    
    # Should have dependencies
    assert len(sentence.dependencies) > 0
    
    # Each dependency should be a tuple of (head, relation, dependent)
    for dep in sentence.dependencies:
        assert len(dep) == 3
        assert isinstance(dep[0], int)  # head index
        assert isinstance(dep[1], str)  # relation type
        assert isinstance(dep[2], int)  # dependent index


def test_language_detection(parser):
    """Test language detection."""
    english_text = "This is an English sentence."
    spanish_text = "Esta es una oración en español."
    
    lang_en = parser.detect_language(english_text)
    lang_es = parser.detect_language(spanish_text)
    
    assert isinstance(lang_en, LanguageCode)
    assert isinstance(lang_es, LanguageCode)


def test_text_preprocessing(parser):
    """Test text preprocessing."""
    text = "  Multiple   spaces   here.  \n\n Extra newlines. "
    cleaned = parser._preprocess_text(text)
    
    assert "   " not in cleaned
    assert cleaned.strip() == cleaned
    assert "\n\n" not in cleaned


def test_batch_parsing(parser):
    """Test batch document parsing."""
    texts = [
        "First document with some text.",
        "Second document with different content.",
        "Third document for testing."
    ]
    doc_ids = ["doc1", "doc2", "doc3"]
    
    documents = parser.parse_batch(texts, doc_ids=doc_ids)
    
    assert len(documents) == 3
    for i, doc in enumerate(documents):
        assert doc.doc_id == doc_ids[i]
        assert len(doc.sentences) > 0


def test_empty_text(parser):
    """Test handling of empty text."""
    text = ""
    document = parser.parse_document(text)
    
    assert isinstance(document, Document)
    assert len(document.sentences) == 0


def test_single_word(parser):
    """Test parsing single word."""
    text = "Hello"
    document = parser.parse_document(text)
    
    assert len(document.sentences) >= 0
    if len(document.sentences) > 0:
        assert len(document.sentences[0].tokens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
