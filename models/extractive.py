"""
Extractive summarization models.
Identifies and ranks salient sentences from documents.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

from .base import BaseSummarizationModel


class ExtractiveSummarizationModel(BaseSummarizationModel):
    """
    Extractive summarization model using sentence ranking.
    Identifies important sentences and returns them in original order.
    """
    
    def load(self) -> None:
        """Load sentence embedding model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def summarize(
        self,
        text: str,
        max_length: int = 128,
        num_sentences: int = 3,
        **kwargs
    ) -> str:
        """
        Extract important sentences from document.
        
        Args:
            text: Input document text
            max_length: Maximum summary length
            num_sentences: Number of sentences to extract
            **kwargs: Additional arguments
            
        Returns:
            Extractive summary with selected sentences
        """
        # Split into sentences
        sentences = text.split(". ")
        
        if len(sentences) <= num_sentences:
            return text
        
        # Encode sentences
        embeddings = self._encode_sentences(sentences)
        
        # Score and select top sentences
        # Placeholder - to be implemented with proper ranking
        selected_indices = list(range(min(num_sentences, len(sentences))))
        
        # Return sentences in original order
        selected_sentences = [
            sentences[i] for i in sorted(selected_indices)
        ]
        
        return ". ".join(selected_sentences) + "."
    
    def batch_summarize(
        self,
        texts: List[str],
        max_length: int = 128,
        num_sentences: int = 3,
        **kwargs
    ) -> List[str]:
        """
        Extract summaries from multiple documents.
        
        Args:
            texts: List of input documents
            max_length: Maximum summary length
            num_sentences: Number of sentences to extract per document
            **kwargs: Additional arguments
            
        Returns:
            List of extractive summaries
        """
        return [
            self.summarize(text, max_length, num_sentences, **kwargs)
            for text in texts
        ]
    
    def _encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            Tensor of sentence embeddings
        """
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return embeddings
