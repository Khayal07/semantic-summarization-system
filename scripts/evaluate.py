"""
Evaluation script for assessing model performance on benchmark datasets.
"""

import argparse
import json
from pathlib import Path
from loguru import logger

from pipelines.summarization import SummarizationPipeline
from datasets.loader import SummarizationDataset
from evaluation.metrics import SummarizationMetrics


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    logger.add(
        f"{log_dir}/evaluation.log",
        rotation="500 MB",
    )


def evaluate(args):
    """
    Evaluate model on dataset.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting evaluation of {args.model}")
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    dataset_loader = SummarizationDataset(args.dataset)
    data = dataset_loader.load(split="validation", max_samples=args.max_samples)
    
    logger.info(f"Loaded {len(data['texts'])} evaluation samples")
    
    # Initialize pipeline
    pipeline = SummarizationPipeline(model_name=args.model)
    pipeline.load_model()
    
    # Generate summaries
    logger.info("Generating summaries...")
    predictions = pipeline.batch_summarize(
        data["texts"],
        max_length=args.max_length,
        min_length=args.min_length,
        batch_size=args.batch_size,
    )
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics_computer = SummarizationMetrics()
    results = metrics_computer.evaluate(
        predictions,
        data["summaries"],
        include_bert_score=args.include_bert_score,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{args.model.replace('/', '_')}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Metrics: {results}")
    
    pipeline.unload_model()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate summarization model")
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--include_bert_score", action="store_true")
    
    args = parser.parse_args()
    
    setup_logging(args.log_dir)
    evaluate(args)


if __name__ == "__main__":
    main()
