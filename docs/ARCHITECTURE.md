# Architecture Documentation

## System Overview

The Semantic Summarization System is built with a modular, layered architecture designed for production deployment and research experimentation. This document outlines the design principles, component interactions, and data flows.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST Server                   │
│                    (api/main.py)                         │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Pipeline Layer                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │ SummarizationPipeline (Orchestration)           │   │
│  │ - Model loading                                  │   │
│  │ - Input preprocessing                           │   │
│  │ - Inference execution                           │   │
│  │ - Post-processing                               │   │
│  └──────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────┐
│ Models │  │ Datasets │
│ Layer  │  │ Layer    │
└────────┘  └──────────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────────────────────────────────────────────┐
│                    Evaluation Layer                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Metrics (ROUGE, BERTScore, BLEU)               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer (`api/`)

**Responsibility**: REST interface for external clients

**Key Components**:
- `main.py`: FastAPI application and route definitions
- `routes/`: Individual endpoint modules
- `schemas.py`: Pydantic request/response models
- `dependencies.py`: Shared dependencies (authentication, logging)

**Key Endpoints**:
- `POST /api/v1/summarize`: Generate summary from text
- `POST /api/v1/batch-summarize`: Process multiple documents
- `GET /api/v1/models`: List available models
- `GET /api/v1/health`: Health check endpoint

**Features**:
- Input validation using Pydantic
- Error handling with standardized responses
- Async request processing
- Request logging and monitoring

### 2. Pipeline Layer (`pipelines/`)

**Responsibility**: Orchestrate end-to-end summarization workflow

**Key Components**:
- `base.py`: Abstract base pipeline class
- `summarization.py`: Main summarization pipeline
- `extractive.py`: Extractive summarization pipeline
- `abstractive.py`: Abstractive summarization pipeline
- `hybrid.py`: Combined approaches

**Pipeline Workflow**:
```
Input → Validation → Preprocessing → Model Selection → Inference → Post-processing → Output
```

**Responsibilities**:
- Load and initialize models
- Preprocess input text (tokenization, truncation)
- Execute inference with appropriate parameters
- Post-process outputs (cleaning, length control)
- Handle errors gracefully

### 3. Models Layer (`models/`)

**Responsibility**: Encapsulate model implementations

**Key Components**:
- `base.py`: Abstract base model class
- `abstractive.py`: Abstractive summarization models
- `extractive.py`: Extractive summarization models
- `utils.py`: Model utilities and helpers

**Supported Architectures**:
- BART (Bidirectional Encoder Representations from Transformers + Autoregressive Decoder)
- T5 (Unified Text-to-Text Transformer)
- PEGASUS (Pre-trained Expert-trained Generative Auto-Regressive Summarization)

**Model Management**:
- Lazy loading for memory efficiency
- Caching of loaded models
- Device management (CPU/GPU)
- Model configuration via Pydantic

### 4. Datasets Layer (`datasets/`)

**Responsibility**: Data loading and preprocessing

**Key Components**:
- `loaders.py`: Dataset loading utilities
- `preprocessors.py`: Text preprocessing
- `collators.py`: Batch collation for training

**Supported Datasets**:
- CNN/DailyMail
- XSum
- SAMSum
- Custom datasets

**Features**:
- Streaming support for large datasets
- Automatic tokenization
- Dynamic padding and batching
- Train/validation/test splits

### 5. Evaluation Layer (`evaluation/`)

**Responsibility**: Assess summary quality

**Key Components**:
- `metrics.py`: ROUGE, BLEU, BERTScore implementations
- `evaluator.py`: Evaluation runner
- `reporters.py`: Results formatting and reporting

**Metrics**:
- **ROUGE**: Lexical overlap metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- **BLEU**: Precision-based n-gram matching
- **BERTScore**: Contextual similarity scoring
- **Length Ratio**: Summary vs. source length analysis

**Evaluation Modes**:
- Single sample evaluation
- Batch evaluation with aggregation
- Comparative analysis across models

### 6. Experiments Layer (`experiments/`)

**Responsibility**: Track and manage experimental runs

**Structure**:
- `configs/`: Experiment configuration files
- `results/`: Experiment outputs
- `logs/`: Experiment logs

**Tracking**:
- Hyperparameter recording
- Model checkpoint management
- Metrics logging
- Comparison across runs

### 7. Utilities Layer (`utils/`)

**Responsibility**: Shared utility functions

**Key Modules**:
- `text_utils.py`: Text manipulation functions
- `torch_utils.py`: PyTorch utilities
- `config.py`: Configuration management
- `logging.py`: Logging setup

## Data Flow

### Summarization Request Flow

```
1. HTTP Request arrives at FastAPI endpoint
   └─ Pydantic validates request schema

2. Request is passed to Pipeline
   ├─ Pipeline loads model (if not cached)
   ├─ Text is preprocessed (tokenized, truncated)
   └─ Model configuration is prepared

3. Model Inference
   ├─ Input ids are generated
   ├─ Model generates summary tokens
   ├─ Beam search (if enabled) creates candidates
   └─ Top summary is selected

4. Post-processing
   ├─ Tokens are decoded to text
   ├─ Length constraints are applied
   ├─ Text cleanup is performed
   └─ Output is formatted

5. Response
   └─ Pydantic serializes response
   └─ HTTP response is sent
```

### Evaluation Flow

```
1. Load generated summaries
2. Load reference summaries
3. Compute metrics
   ├─ ROUGE (lexical overlap)
   ├─ BERTScore (semantic similarity)
   └─ BLEU (precision-based)
4. Aggregate results
5. Generate report
```

## Configuration Management

### Configuration Hierarchy

```
1. Default Values (in code)
   ↓
2. YAML Config Files (configs/)
   ↓
3. Environment Variables (.env)
   ↓
4. Runtime Overrides (API parameters)
```

### Key Configuration Files

- `configs/models.yaml`: Model definitions
- `configs/inference.yaml`: Inference settings
- `configs/training.yaml`: Training hyperparameters
- `configs/evaluation.yaml`: Evaluation settings

## Error Handling

### Error Strategy

1. **Input Validation**: Pydantic schemas catch malformed requests
2. **Runtime Errors**: Try-except blocks with specific error types
3. **Model Errors**: Fallback models or graceful degradation
4. **Resource Errors**: Memory management and timeouts

### Error Response Format

```json
{
  "status": "error",
  "error_code": "MODEL_LOADING_FAILED",
  "message": "Failed to load model: facebook/bart-large-cnn",
  "details": {
    "original_error": "...",
    "suggestions": ["Check model name", "Verify GPU availability"]
  }
}
```

## Performance Considerations

### Optimization Strategies

1. **Model Caching**: Models are loaded once and reused
2. **Batch Processing**: Multiple documents processed together
3. **GPU Optimization**: Efficient tensor operations with CUDA
4. **Memory Management**: Gradient checkpointing and mixed precision
5. **Async Processing**: Non-blocking I/O for API endpoints

### Scalability

- **Horizontal**: Multiple API instances behind load balancer
- **Vertical**: GPU memory optimization for larger models
- **Distributed**: Distributed inference with Ray or similar

## Testing Strategy

### Test Pyramid

```
         ▲
        ╱ ╲
       ╱   ╲  End-to-End Tests
      ╱─────╲
     ╱       ╲
    ╱─────────╲ Integration Tests
   ╱           ╲
  ╱─────────────╲ Unit Tests
 ╱_______________╲
```

### Test Categories

- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **API Tests**: Endpoint functionality testing
- **Performance Tests**: Latency and throughput benchmarking

## Deployment Architecture

### Development

```
Local Machine
├── Python venv
├── HuggingFace cache (~1-10GB)
├── FastAPI dev server
└── GPU (optional)
```

### Production

```
Cloud Infrastructure
├── Container (Docker)
├── Orchestration (Kubernetes)
├── Load Balancer
├── Model Cache Volume
├── Monitoring & Logging
└── High-performance GPUs
```

## Future Architectural Enhancements

1. **Model Serving**: TorchServe or TensorFlow Serving for optimized inference
2. **Distributed Training**: Multi-GPU and multi-node training with Distributed Data Parallel (DDP)
3. **Caching Layer**: Redis for response caching
4. **Message Queue**: Celery for async task processing
5. **Database**: PostgreSQL for experiment tracking
6. **Monitoring**: Prometheus + Grafana for metrics and visualization
7. **A/B Testing**: Experimentation framework for model comparison
8. **Feature Store**: Centralized feature management for training

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Web Framework** | FastAPI + Uvicorn |
| **Deep Learning** | PyTorch |
| **Transformers** | Hugging Face Transformers |
| **Data Processing** | Hugging Face Datasets, Pandas |
| **Evaluation** | rouge-score, BERTScore |
| **Logging** | Loguru |
| **Configuration** | Pydantic, YAML |
| **Testing** | Pytest |
| **Code Quality** | Black, Flake8, MyPy |

## Design Patterns Used

1. **Factory Pattern**: Model creation in `models/base.py`
2. **Strategy Pattern**: Different summarization strategies (abstractive, extractive)
3. **Pipeline Pattern**: Orchestration of sequential steps
4. **Singleton Pattern**: Model caching
5. **Dependency Injection**: FastAPI dependencies for middleware

---

**Last Updated**: May 2024  
**Version**: 1.0
