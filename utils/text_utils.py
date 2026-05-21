"""
Text processing utilities.
Provides helper functions for text manipulation and preprocessing.
"""

from typing import List, Tuple
import re


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting - can be improved with NLTK
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length in characters
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def get_text_stats(text: str) -> dict:
    """
    Calculate text statistics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    sentences = split_sentences(text)
    words = text.split()
    
    return {
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_chars": len(text),
        "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
    }
