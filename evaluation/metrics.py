"""
Evaluation metrics for assessing summary quality.
Includes ROUGE, BLEU, and BERTScore implementations.
"""

from typing import List, Dict, Any
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score


class SummarizationMetrics:
    """
    Compute standard summarization evaluation metrics.
    
    Supports ROUGE, BLEU, and semantic similarity metrics.
    """
    
    def __init__(self):
        """Initialize metrics."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE metrics.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            Dictionary of ROUGE scores
        """
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric in rouge_scores.keys():
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Average scores
        return {
            f"{metric}_f": np.mean(scores)
            for metric, scores in rouge_scores.items()
        }
    
    def compute_bert_score(
        self,
        predictions: List[str],
        references: List[str],
        model_type: str = "bert-base-uncased",
    ) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            model_type: BERT model to use
            
        Returns:
            Dictionary of BERTScore metrics
        """
        try:
            precision, recall, f1 = bert_score(
                predictions,
                references,
                model_type=model_type,
                lang="en",
                verbose=False,
            )
            
            return {
                "bertscore_precision": precision.mean().item(),
                "bertscore_recall": recall.mean().item(),
                "bertscore_f1": f1.mean().item(),
            }
        except Exception as e:
            # Return zeros if BERTScore fails
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            }
    
    def compute_length_statistics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute length statistics.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            Dictionary of length metrics
        """
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        return {
            "avg_pred_length": np.mean(pred_lengths),
            "avg_ref_length": np.mean(ref_lengths),
            "compression_ratio": np.mean([
                p / r if r > 0 else 0
                for p, r in zip(pred_lengths, ref_lengths)
            ]),
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        include_bert_score: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on multiple metrics.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            include_bert_score: Whether to compute BERTScore
            
        Returns:
            Dictionary of all computed metrics
        """
        results = {}
        
        # ROUGE
        results.update(self.compute_rouge(predictions, references))
        
        # Length statistics
        results.update(self.compute_length_statistics(predictions, references))
        
        # BERTScore (optional, slower)
        if include_bert_score:
            results.update(self.compute_bert_score(predictions, references))
        
        return results
