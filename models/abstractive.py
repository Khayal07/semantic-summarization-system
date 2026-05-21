"""
Abstractive summarization models using sequence-to-sequence architectures.
Supports BART, T5, and PEGASUS models.
"""

from typing import Optional, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
)

from .base import BaseSummarizationModel


class AbstractiveSummarizationModel(BaseSummarizationModel):
    """
    Abstractive summarization model using encoder-decoder architectures.
    Supports BART, T5, and PEGASUS models from HuggingFace.
    """
    
    def load(self) -> None:
        """Load BART/T5/PEGASUS model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def summarize(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 30,
        num_beams: int = 4,
        early_stopping: bool = True,
        **kwargs
    ) -> str:
        """
        Generate abstractive summary using seq2seq model.
        
        Args:
            text: Input document text
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early
            **kwargs: Additional generation arguments
            
        Returns:
            Generated abstractive summary
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                **kwargs
            )
        
        # Decode summary
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
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
        Generate abstractive summaries for batch of texts.
        
        Args:
            texts: List of input documents
            max_length: Maximum summary length
            min_length: Minimum summary length
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summaries
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    **kwargs
                )
            
            # Decode summaries
            batch_summaries = self.tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True
            )
            summaries.extend(batch_summaries)
        
        return summaries
