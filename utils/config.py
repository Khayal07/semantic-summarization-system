"""
Configuration management using Pydantic.
Handles environment variables, YAML configs, and runtime settings.
"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os


class ModelConfig(BaseModel):
    """Model configuration."""
    
    name: str = Field(default="facebook/bart-large-cnn", description="Model name")
    model_type: str = Field(default="bart", description="Model type")
    device: str = Field(default="cuda", description="Device (cuda/cpu)")
    max_length: int = Field(default=128, description="Maximum output length")
    min_length: int = Field(default=30, description="Minimum output length")


class InferenceConfig(BaseModel):
    """Inference configuration."""
    
    batch_size: int = Field(default=16, description="Batch size")
    num_beams: int = Field(default=4, description="Number of beams")
    early_stopping: bool = Field(default=True, description="Early stopping")
    length_penalty: float = Field(default=2.0, description="Length penalty")


class APIConfig(BaseSettings):
    """API configuration from environment variables."""
    
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # API
    api: APIConfig = APIConfig()
    
    # Models
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    
    # Paths
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint directory")
    model_cache_dir: str = Field(default="./model_cache", description="Model cache directory")
    log_dir: str = Field(default="./logs", description="Log directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create global application configuration."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    global _config
    if config_path and os.path.exists(config_path):
        # TODO: Load from YAML
        pass
    _config = AppConfig()
    return _config
