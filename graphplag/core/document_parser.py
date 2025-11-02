"""
Document Parser Module

Handles document preprocessing, tokenization, sentence segmentation,
and language detection using spaCy.
"""

from typing import List, Optional, Dict
import re
import spacy
from spacy.language import Language
from langdetect import detect, LangDetectException

from graphplag.core.models import Document, Sentence, LanguageCode


class DocumentParser:
    """
    Parse raw text documents into structured Document objects.
    
    Uses spaCy for linguistic analysis including tokenization, POS tagging,
    lemmatization, and dependency parsing.
    """
    
    def __init__(
        self,
        language: str = "en",
        model_name: Optional[str] = None,
        disable: Optional[List[str]] = None
    ):
        """
        Initialize the DocumentParser.
        
        Args:
            language: Language code (default: 'en')
            model_name: Specific spaCy model to use (e.g., 'en_core_web_trf')
            disable: Pipeline components to disable for performance
        """
        self.language = language
        self.model_name = model_name or self._get_default_model(language)
        self.disable = disable or []
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.model_name, disable=self.disable)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{self.model_name}' not found. "
                f"Please install it with: python -m spacy download {self.model_name}"
            )
        
        # Set max length for large documents
        self.nlp.max_length = 2000000
    
    @staticmethod
    def _get_default_model(language: str) -> str:
        """Get default spaCy model for a language."""
        model_map = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm",
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
            "pt": "pt_core_news_sm",
            "nl": "nl_core_news_sm",
            "ru": "ru_core_news_sm",
            "zh": "zh_core_web_sm",
            "ja": "ja_core_news_sm",
        }
        return model_map.get(language, "en_core_web_sm")
    
    def parse_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        auto_detect_language: bool = False
    ) -> Document:
        """
        Parse a document from raw text.
        
        Args:
            text: Raw text to parse
            doc_id: Optional document identifier
            auto_detect_language: Whether to detect language automatically
            
        Returns:
            Document object with parsed sentences
        """
        # Clean text
        text = self._preprocess_text(text)
        
        # Detect language if requested
        detected_lang = LanguageCode.UNKNOWN
        if auto_detect_language:
            detected_lang = self.detect_language(text)
            
            # If detected language differs from parser language, warn user
            if detected_lang.value != self.language and detected_lang != LanguageCode.UNKNOWN:
                print(f"Warning: Detected language '{detected_lang.value}' differs from parser language '{self.language}'")
        else:
            try:
                detected_lang = LanguageCode(self.language)
            except ValueError:
                detected_lang = LanguageCode.UNKNOWN
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = self.extract_sentences(doc)
        
        # Create Document object
        document = Document(
            text=text,
            sentences=sentences,
            language=detected_lang,
            doc_id=doc_id,
            metadata={
                "num_sentences": len(sentences),
                "num_tokens": len(doc),
                "parser_model": self.model_name
            }
        )
        
        return document
    
    def extract_sentences(self, doc: Language) -> List[Sentence]:
        """
        Extract sentences from a spaCy Doc object.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of Sentence objects
        """
        sentences = []
        
        for idx, sent in enumerate(doc.sents):
            # Extract tokens and linguistic features
            tokens = [token.text for token in sent]
            lemmas = [token.lemma_ for token in sent]
            pos_tags = [token.pos_ for token in sent]
            
            # Extract dependencies within the sentence
            dependencies = []
            for token in sent:
                if token.head != token:  # Not root
                    # Get indices relative to sentence start
                    head_idx = token.head.i - sent.start
                    dep_idx = token.i - sent.start
                    dependencies.append((head_idx, token.dep_, dep_idx))
            
            sentence = Sentence(
                text=sent.text.strip(),
                index=idx,
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                dependencies=dependencies
            )
            
            sentences.append(sentence)
        
        return sentences
    
    def detect_language(self, text: str) -> LanguageCode:
        """
        Detect the language of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageCode enum value
        """
        try:
            lang_code = detect(text)
            # Map to our LanguageCode enum
            try:
                return LanguageCode(lang_code)
            except ValueError:
                return LanguageCode.UNKNOWN
        except LangDetectException:
            return LanguageCode.UNKNOWN
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def parse_batch(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> List[Document]:
        """
        Parse multiple documents efficiently.
        
        Args:
            texts: List of raw texts
            doc_ids: Optional list of document identifiers
            batch_size: Batch size for processing
            
        Returns:
            List of Document objects
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]
        
        documents = []
        
        # Preprocess all texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        # Process in batches
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            
            # Use spaCy's pipe for efficient batch processing
            for doc, doc_id in zip(self.nlp.pipe(batch_texts), batch_ids):
                sentences = self.extract_sentences(doc)
                
                document = Document(
                    text=doc.text,
                    sentences=sentences,
                    language=LanguageCode(self.language) if self.language in [e.value for e in LanguageCode] else LanguageCode.UNKNOWN,
                    doc_id=doc_id,
                    metadata={
                        "num_sentences": len(sentences),
                        "num_tokens": len(doc),
                        "parser_model": self.model_name
                    }
                )
                
                documents.append(document)
        
        return documents
    
    def __repr__(self) -> str:
        return f"DocumentParser(language='{self.language}', model='{self.model_name}')"
