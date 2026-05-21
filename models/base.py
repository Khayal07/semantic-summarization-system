"""
Base model class for summarization models.
Provides interface for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import torch


class BaseSummarizationModel(ABC):
    """
    Abstract base class for summarization models.
    
    All concrete model implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the base model.
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to load model on (cuda/cpu)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = kwargs
        
    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        pass
    
    @abstractmethod
    def summarize(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 30,
        **kwargs
    ) -> str:
        """
        Generate a summary for the input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            **kwargs: Additional generation arguments
            
        Returns:
            Generated summary string
        """
        pass
    
    @abstractmethod
    def batch_summarize(
        self,
        texts: List[str],
        max_length: int = 128,
        min_length: int = 30,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum length of generated summaries
            min_length: Minimum length of generated summaries
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated summaries
        """
        pass
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
    
    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
            self.device = device
