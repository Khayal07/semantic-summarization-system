"""
FastAPI application definition.
REST API endpoints for summarization service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger

from utils.config import get_config
from pipelines.summarization import SummarizationPipeline

# Initialize logger
logger.remove()
logger.add("logs/api.log", rotation="500 MB", retention="7 days")
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
)

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Summarization System",
    description="Production-grade NLP system for abstractive and extractive text summarization",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SummarizeRequest(BaseModel):
    """Summarization request model."""
    
    text: str = Field(..., description="Text to summarize", min_length=10)
    max_length: int = Field(default=128, description="Max summary length", ge=30)
    min_length: int = Field(default=30, description="Min summary length", le=128)
    model: str = Field(
        default="facebook/bart-large-cnn",
        description="Model to use"
    )


class SummarizeResponse(BaseModel):
    """Summarization response model."""
    
    original_text: str
    summary: str
    model_used: str
    input_length: int
    summary_length: int
    compression_ratio: float


class BatchSummarizeRequest(BaseModel):
    """Batch summarization request model."""
    
    texts: List[str] = Field(..., description="Texts to summarize", min_items=1)
    max_length: int = Field(default=128, description="Max summary length")
    min_length: int = Field(default=30, description="Min summary length")
    model: str = Field(default="facebook/bart-large-cnn", description="Model to use")


class BatchSummarizeResponse(BaseModel):
    """Batch summarization response model."""
    
    summaries: List[SummarizeResponse]
    processing_time: float


# Pipeline instance
_pipeline: Optional[SummarizationPipeline] = None


def get_pipeline() -> SummarizationPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        config = get_config()
        _pipeline = SummarizationPipeline(
            model_name=config.model.name,
            model_type="abstractive",
            device=config.model.device,
        )
        _pipeline.load_model()
    return _pipeline


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Semantic Summarization System",
        "version": "0.1.0"
    }


# Summarization endpoints
@app.post("/api/v1/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request: SummarizeRequest):
    """
    Generate summary for input text.
    
    Args:
        request: Summarization request with text and parameters
        
    Returns:
        Summary response with generated summary and metadata
    """
    try:
        logger.info(f"Summarization request received for {len(request.text)} characters")
        
        pipeline = get_pipeline()
        summary = pipeline.summarize(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
        )
        
        compression_ratio = len(summary) / len(request.text) if request.text else 0
        
        return SummarizeResponse(
            original_text=request.text,
            summary=summary,
            model_used=request.model,
            input_length=len(request.text),
            summary_length=len(summary),
            compression_ratio=compression_ratio,
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Model information
@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """List available models."""
    return {
        "available_models": [
            {
                "name": "facebook/bart-large-cnn",
                "type": "abstractive",
                "description": "BART model fine-tuned on CNN/DailyMail",
            },
            {
                "name": "google/t5-base",
                "type": "abstractive",
                "description": "T5 base model for text-to-text tasks",
            },
            {
                "name": "google/pegasus-large",
                "type": "abstractive",
                "description": "PEGASUS model specialized for summarization",
            },
        ]
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Semantic Summarization System")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.unload_model()
    logger.info("Shutting down Semantic Summarization System")


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.api.api_host,
        port=config.api.api_port,
        log_level="info",
    )
