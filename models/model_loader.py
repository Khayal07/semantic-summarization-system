"""
Model loading and management utilities.
Handles model initialization, caching, and device management.
"""

from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from loguru import logger


class ModelLoader:
    """
    Utility class for loading and managing transformer models.
    
    Features:
    - Lazy loading for memory efficiency
    - Model caching
    - Device management (CPU/GPU)
    - Tokenizer management
    """
    
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize model loader.
        
        Args:
            device: Device to load models on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self._logger = logger.bind(component="model_loader")
        
        self._logger.info(f"Initialized ModelLoader with device: {self.device}")
    
    @classmethod
    def get_device(cls) -> str:
        """Get available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """
        Load a pretrained model.
        
        Args:
            model_name: Name of model from Hugging Face
            force_reload: Whether to force reload even if cached
            
        Returns:
            Loaded model
        """
        # Check cache
        if model_name in self._model_cache and not force_reload:
            self._logger.info(f"Loading model from cache: {model_name}")
            return self._model_cache[model_name]
        
        try:
            self._logger.info(f"Loading model: {model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to(self.device)
            
            # Cache the model
            self._model_cache[model_name] = model
            
            self._logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            self._logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_tokenizer(self, model_name: str, force_reload: bool = False):
        """
        Load a pretrained tokenizer.
        
        Args:
            model_name: Name of model from Hugging Face
            force_reload: Whether to force reload even if cached
            
        Returns:
            Loaded tokenizer
        """
        # Check cache
        if model_name in self._tokenizer_cache and not force_reload:
            self._logger.info(f"Loading tokenizer from cache: {model_name}")
            return self._tokenizer_cache[model_name]
        
        try:
            self._logger.info(f"Loading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Cache the tokenizer
            self._tokenizer_cache[model_name] = tokenizer
            
            self._logger.info(f"Successfully loaded tokenizer: {model_name}")
            return tokenizer
            
        except Exception as e:
            self._logger.error(f"Failed to load tokenizer {model_name}: {e}")
            raise
    
    def get_model_and_tokenizer(self, model_name: str):
        """
        Get both model and tokenizer.
        
        Args:
            model_name: Name of model from Hugging Face
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        return model, tokenizer
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models and tokenizers."""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of model from Hugging Face
            
        Returns:
            Dictionary with model information
        """
        return {
            "name": model_name,
            "cached": model_name in cls._model_cache,
            "tokenizer_cached": model_name in cls._tokenizer_cache,
        }


# Global model loader instance
_loader: Optional[ModelLoader] = None


def get_model_loader(device: str = "cuda") -> ModelLoader:
    """Get or create global model loader instance."""
    global _loader
    if _loader is None:
        _loader = ModelLoader(device=device)
    return _loader


def load_model(model_name: str, device: str = "cuda"):
    """Convenience function to load a model."""
    loader = get_model_loader(device)
    return loader.load_model(model_name)


def load_tokenizer(model_name: str, device: str = "cuda"):
    """Convenience function to load a tokenizer."""
    loader = get_model_loader(device)
    return loader.load_tokenizer(model_name)
