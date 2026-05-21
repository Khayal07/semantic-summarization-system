"""
Main entry point for Semantic Summarization System.
Can be run directly or imported as a module.
"""

import logging
from pathlib import Path

from loguru import logger

from utils.config import get_config


def setup_logging():
    """Initialize logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.remove()  # Remove default handler
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="7 days",
        level="INFO",
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def main():
    """Main application entry point."""
    setup_logging()
    
    logger.info("Starting Semantic Summarization System")
    
    # Load configuration
    config = get_config()
    logger.info(f"Configuration loaded: {config.model.name}")
    
    # Import and run API
    from api.main import app
    import uvicorn
    
    logger.info(f"Starting API server on {config.api.api_host}:{config.api.api_port}")
    
    uvicorn.run(
        app,
        host=config.api.api_host,
        port=config.api.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
