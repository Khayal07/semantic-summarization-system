"""
Text preprocessing utilities for NLP pipeline.
Handles text cleaning, normalization, and tokenization for summarization tasks.

This module provides production-grade text preprocessing functionality with:
- URL and email removal
- Whitespace normalization
- Special character handling
- Sentence segmentation
- Token management
"""

import re
from typing import List, Optional
from loguru import logger

# Optional NLTK support for advanced tokenization
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    HAS_NLTK = True
except (ImportError, LookupError):
    HAS_NLTK = False


class TextPreprocessor:
    """
    Production-grade text preprocessing for NLP summarization pipelines.
    
    Provides modular text preprocessing methods including:
    - clean_text(): Full text cleaning pipeline
    - normalize_whitespace(): Normalize spaces and line breaks
    - remove_special_characters(): Remove or normalize special characters
    - segment_sentences(): Split text into sentences
    
    Attributes:
        lowercase: Whether to convert text to lowercase
        keep_punctuation: Whether to preserve basic punctuation marks
    """
    
    # Preserved punctuation marks
    PRESERVED_PUNCTUATION = r'\.\!\?\-\,\:\;'
    
    def __init__(self, lowercase: bool = False, keep_punctuation: bool = True) -> None:
        """
        Initialize the TextPreprocessor.
        
        Args:
            lowercase: Convert all text to lowercase. Defaults to False.
            keep_punctuation: Preserve basic punctuation. Defaults to True.
        """
        self.lowercase = lowercase
        self.keep_punctuation = keep_punctuation
        self._logger = logger.bind(component="TextPreprocessor")
        
        self._logger.debug(
            f"Initialized TextPreprocessor (lowercase={lowercase}, "
            f"keep_punctuation={keep_punctuation})"
        )
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove or normalize special characters in text.
        
        Removes URLs, email addresses, and non-alphanumeric characters
        while optionally preserving basic punctuation.
        
        Args:
            text: Input text to clean
            
        Returns:
            Text with special characters removed or normalized
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> text = "Check this: https://example.com! Also: test@email.com"
            >>> preprocessor.remove_special_characters(text)
            'Check this:  Also: '
        """
        if not text:
            return ""
        
        # Remove URLs (http/https/www)
        text = re.sub(r'http\S+|https\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove or keep special characters based on configuration
        if self.keep_punctuation:
            # Keep only alphanumeric, whitespace, and basic punctuation
            pattern = f'[^\\w\\s{self.PRESERVED_PUNCTUATION}]'
            text = re.sub(pattern, '', text)
        else:
            # Remove all non-alphanumeric characters except whitespace
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Consolidates multiple spaces, tabs, and newlines into single spaces.
        Removes leading and trailing whitespace.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Text with normalized whitespace
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> text = "  Hello    world  \\n\\t  test  "
            >>> preprocessor.normalize_whitespace(text)
            'Hello world test'
        """
        if not text:
            return ""
        
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading and trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str, lowercase: Optional[bool] = None) -> str:
        """
        Perform complete text cleaning pipeline.
        
        Orchestrates all preprocessing steps in sequence:
        1. Remove special characters (URLs, emails, etc.)
        2. Normalize whitespace
        3. Optional: Convert to lowercase
        
        Args:
            text: Input text to clean
            lowercase: Override instance lowercase setting. If None, uses instance setting.
            
        Returns:
            Fully cleaned and normalized text
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> text = "  Visit https://example.com for info!  "
            >>> preprocessor.clean_text(text)
            'Visit for info !'
        """
        if not text:
            self._logger.warning("Received empty text for cleaning")
            return ""
        
        # Determine whether to apply lowercase
        apply_lowercase = lowercase if lowercase is not None else self.lowercase
        
        # Step 1: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 2: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 3: Convert to lowercase if needed
        if apply_lowercase:
            text = text.lower()
        
        self._logger.debug(f"Cleaned text: {len(text)} characters")
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Uses NLTK if available for reliable sentence segmentation,
        falls back to period-based splitting if NLTK is unavailable.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of sentences (non-empty, stripped)
        """
        if not text:
            return []
        
        try:
            if HAS_NLTK:
                sentences = nltk.sent_tokenize(text)
            else:
                # Fallback: split by common sentence endings
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Filter empty sentences and strip whitespace
            sentences = [s.strip() for s in sentences if s.strip()]
            self._logger.debug(f"Segmented text into {len(sentences)} sentences")
            return sentences
            
        except Exception as e:
            self._logger.error(f"Sentence segmentation failed: {e}")
            return [text.strip()]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Splits text into individual word tokens. Uses NLTK if available,
        otherwise uses simple whitespace splitting.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        if not text:
            return []
        
        try:
            if HAS_NLTK:
                tokens = nltk.word_tokenize(text)
            else:
                # Simple whitespace tokenization fallback
                tokens = text.split()
            
            return tokens
            
        except Exception as e:
            self._logger.error(f"Tokenization failed: {e}")
            return text.split()
    
    def preprocess(
        self,
        text: str,
        max_length: int = 1024,
        clean: bool = True,
    ) -> str:
        """
        Complete preprocessing pipeline.
        
        Orchestrates the full preprocessing workflow:
        1. Clean text (if enabled)
        2. Truncate to max_length if necessary
        
        Args:
            text: Input text to preprocess
            max_length: Maximum text length in characters
            clean: Whether to apply full cleaning pipeline
            
        Returns:
            Fully preprocessed text
        """
        if not text:
            return ""
        
        # Step 1: Clean if requested
        if clean:
            text = self.clean_text(text)
        else:
            # At minimum, normalize whitespace
            text = self.normalize_whitespace(text)
        
        # Step 2: Truncate if necessary
        if len(text) > max_length:
            # Truncate at word boundary
            truncated = text[:max_length]
            # Find last space to avoid cutting words
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.9:  # Only if space is close
                text = truncated[:last_space] + '...'
            else:
                text = truncated + '...'
        
        return text


# ============================================================================
# Module-level convenience functions for quick access
# ============================================================================

_preprocessor: Optional[TextPreprocessor] = None


def get_preprocessor(lowercase: bool = False) -> TextPreprocessor:
    """
    Get or create global preprocessor instance.
    
    Args:
        lowercase: Initialize with lowercase setting
        
    Returns:
        TextPreprocessor instance
    """
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor(lowercase=lowercase)
    return _preprocessor


def clean_text(text: str, lowercase: bool = False) -> str:
    """
    Convenience function: Clean and normalize text.
    
    Args:
        text: Input text to clean
        lowercase: Whether to apply lowercase conversion
        
    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor(lowercase=lowercase)
    return preprocessor.clean_text(text)


def normalize_whitespace(text: str) -> str:
    """
    Convenience function: Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    return get_preprocessor().normalize_whitespace(text)


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Convenience function: Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: Whether to preserve basic punctuation
        
    Returns:
        Text with special characters removed
    """
    preprocessor = TextPreprocessor(keep_punctuation=keep_punctuation)
    return preprocessor.remove_special_characters(text)


def segment_sentences(text: str) -> List[str]:
    """
    Convenience function: Segment text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    return get_preprocessor().segment_sentences(text)


def tokenize(text: str) -> List[str]:
    """
    Convenience function: Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    return get_preprocessor().tokenize(text)


def preprocess(
    text: str,
    max_length: int = 1024,
    clean: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Convenience function: Complete preprocessing pipeline.
    
    Args:
        text: Input text to preprocess
        max_length: Maximum text length in characters
        clean: Whether to apply full cleaning
        lowercase: Whether to apply lowercase conversion
        
    Returns:
        Fully preprocessed text
    """
    preprocessor = TextPreprocessor(lowercase=lowercase)
    return preprocessor.preprocess(text, max_length, clean)
