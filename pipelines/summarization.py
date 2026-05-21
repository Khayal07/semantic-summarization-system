"""
Main summarization pipeline orchestrating end-to-end workflow.
Handles model loading, inference, and post-processing.
"""

from typing import Optional, List
from loguru import logger

from models.abstractive import AbstractiveSummarizationModel
from models.extractive import ExtractiveSummarizationModel
from utils.text_utils import clean_text, truncate_text


class SummarizationPipeline:
    """
    End-to-end summarization pipeline.
    
    Orchestrates model loading, preprocessing, inference, and post-processing
    for generating document summaries.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        model_type: str = "abstractive",
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize summarization pipeline.
        
        Args:
            model_name: Name of pre-trained model
            model_type: Type of summarization (abstractive/extractive)
            device: Device to run on (cuda/cpu)
            **kwargs: Additional model arguments
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.model = None
        self._model_kwargs = kwargs
        self._logger = logger.bind(pipeline="summarization")
        
        self._logger.info(
            f"Initializing {model_type} pipeline with {model_name}"
        )
    
    def load_model(self) -> None:
        """Load the summarization model."""
        try:
            if self.model_type == "abstractive":
                self.model = AbstractiveSummarizationModel(
                    self.model_name,
                    device=self.device,
                    **self._model_kwargs
                )
            elif self.model_type == "extractive":
                self.model = ExtractiveSummarizationModel(
                    self.model_name,
                    device=self.device,
                    **self._model_kwargs
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model.load()
            self._logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            self._logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def summarize(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 30,
        clean_input: bool = True,
        **kwargs
    ) -> str:
        """
        Generate summary for input text.
        
        Args:
            text: Input document text
            max_length: Maximum summary length
            min_length: Minimum summary length
            clean_input: Whether to clean input text
            **kwargs: Additional generation arguments
            
        Returns:
            Generated summary
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        if clean_input:
            text = clean_text(text)
        
        text = truncate_text(text, max_length=512)
        
        # Inference
        self._logger.debug(f"Generating summary for {len(text)} characters")
        summary = self.model.summarize(
            text,
            max_length=max_length,
            min_length=min_length,
            **kwargs
        )
        
        # Post-processing
        summary = clean_text(summary)
        
        self._logger.debug(f"Generated summary: {len(summary)} characters")
        return summary
    
    def batch_summarize(
        self,
        texts: List[str],
        max_length: int = 128,
        min_length: int = 30,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input documents
            max_length: Maximum summary length
            min_length: Minimum summary length
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated summaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self._logger.info(f"Batch summarizing {len(texts)} documents")
        
        summaries = self.model.batch_summarize(
            texts,
            max_length=max_length,
            min_length=min_length,
            batch_size=batch_size,
            **kwargs
        )
        
        return summaries
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            self.model.unload()
            self.model = None
            self._logger.info("Model unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
