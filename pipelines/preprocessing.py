"""
Text preprocessing utilities for NLP pipeline.
Handles text cleaning, tokenization, and normalization.
"""

import re
import nltk
from typing import List, Tuple
from loguru import logger

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """
    Text preprocessing and tokenization utilities.
    
    Handles:
    - Text cleaning and normalization
    - Sentence segmentation
    - Token length management
    """
    
    def __init__(self, lowercase: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
        """
        self.lowercase = lowercase
        self._logger = logger.bind(component="preprocessor")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.\!\?\-\,\:\;]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self._logger.error(f"Sentence segmentation failed: {e}")
            # Fallback: split by periods
            return [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return nltk.word_tokenize(text)
    
    def preprocess(
        self,
        text: str,
        max_length: int = 1024,
        clean: bool = True,
    ) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            max_length: Maximum text length (in characters)
            clean: Whether to clean text
            
        Returns:
            Preprocessed text
        """
        if clean:
            text = self.clean_text(text)
        
        # Truncate if necessary
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
        
        return text


# Module-level convenience functions
_preprocessor = None


def get_preprocessor() -> TextPreprocessor:
    """Get or create global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor


def clean_text(text: str, lowercase: bool = False) -> str:
    """Clean and normalize text."""
    preprocessor = TextPreprocessor(lowercase=lowercase)
    return preprocessor.clean_text(text)


def segment_sentences(text: str) -> List[str]:
    """Segment text into sentences."""
    return get_preprocessor().segment_sentences(text)


def preprocess(text: str, max_length: int = 1024, clean: bool = True) -> str:
    """Complete preprocessing pipeline."""
    return get_preprocessor().preprocess(text, max_length, clean)
