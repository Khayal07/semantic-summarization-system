"""
Inference script for generating summaries from command line.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

from pipelines.summarization import SummarizationPipeline
from utils.text_utils import get_text_stats


def setup_logging():
    """Setup logging."""
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/inference.log", rotation="500 MB")


def inference(args):
    """
    Run inference on input text.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Loading model: {args.model}")
    
    # Initialize pipeline
    pipeline = SummarizationPipeline(
        model_name=args.model,
        model_type=args.model_type,
    )
    pipeline.load_model()
    
    # Read input
    if args.input_file:
        logger.info(f"Reading from {args.input_file}")
        with open(args.input_file, "r") as f:
            text = f.read()
    else:
        logger.info("Reading from stdin...")
        text = sys.stdin.read()
    
    logger.info(f"Input text length: {len(text)} characters")
    
    # Display input statistics
    if args.verbose:
        stats = get_text_stats(text)
        logger.info(f"Input statistics: {stats}")
    
    # Generate summary
    logger.info("Generating summary...")
    summary = pipeline.summarize(
        text,
        max_length=args.max_length,
        min_length=args.min_length,
    )
    
    # Output results
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary)
    print("="*80 + "\n")
    
    if args.verbose:
        logger.info(f"Summary length: {len(summary)} characters")
        compression = len(summary) / len(text) if text else 0
        logger.info(f"Compression ratio: {compression:.2%}")
    
    # Save if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(summary)
        logger.info(f"Summary saved to {args.output_file}")
    
    pipeline.unload_model()
    logger.info("Inference complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate text summaries")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input text file (default: stdin)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for summary",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        help="Model to use",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="abstractive",
        choices=["abstractive", "extractive"],
        help="Type of summarization",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum summary length",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Minimum summary length",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    inference(args)


if __name__ == "__main__":
    main()
