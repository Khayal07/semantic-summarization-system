# Semantic Summarization System

A production-grade NLP system for abstractive and extractive text summarization using state-of-the-art transformer models. This project integrates modern deep learning architectures with scalable inference pipelines for high-quality document summarization.

## Overview

The Semantic Summarization System provides:

- **Abstractive Summarization**: Generate fluent, semantically compressed summaries using transformer-based seq2seq models
- **Extractive Summarization**: Identify and rank salient sentences from source documents
- **Hybrid Approaches**: Combine extractive and abstractive methods for optimal results
- **REST API**: Production-ready FastAPI endpoints for inference at scale
- **Evaluation Framework**: Comprehensive metrics (ROUGE, BLEU, BERTScore) for model assessment
- **Experiment Tracking**: Structured pipeline for hyperparameter tuning and model comparison

## Objectives

1. **Model Exploration**: Evaluate state-of-the-art pre-trained models (T5, BART, PEGASUS)
2. **Domain Adaptation**: Fine-tune models on specialized corpora (news, scientific, legal documents)
3. **Performance Optimization**: Optimize inference latency and throughput for production deployment
4. **Evaluation Excellence**: Implement multi-faceted evaluation using automated metrics and human assessment
5. **Scalability**: Build pipelines that handle variable document lengths and batch processing

## Planned NLP Pipeline

```
Raw Documents
    ↓
Preprocessing (Tokenization, Normalization)
    ↓
Model Selection (Abstractive/Extractive)
    ↓
Inference (Sequence Generation)
    ↓
Post-processing (Cleaning, Length Control)
    ↓
Evaluation (ROUGE, BERTScore, Human Assessment)
    ↓
Output Summaries
```

## Model Candidates

### Abstractive Models

- **BART** (`facebook/bart-large-cnn`)
  - Bidirectional encoder + autoregressive decoder
  - Pre-trained on denoising task
  - Strong performance on news summarization

- **T5** (`google/t5-base`, `google/t5-large`)
  - Unified text-to-text transformer
  - Flexible task formulation
  - Excellent transfer learning capabilities

- **PEGASUS** (`google/pegasus-large`)
  - Pre-trained with gap-sentences objectives
  - Specialized for abstractive summarization
  - State-of-the-art on multiple benchmarks

### Extractive Models

- **BERT-based Ranking**: Fine-tuned BERT for sentence relevance scoring
- **Fine-tuned RoBERTa**: For domain-specific extractive tasks

## Evaluation Metrics

### Automatic Metrics

- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence

- **BLEU** (Bilingual Evaluation Understudy)
  - Precision-based n-gram matching
  - Penalizes brevity

- **BERTScore**
  - Contextual similarity using BERT embeddings
  - Better correlation with human judgment
  - Context-aware evaluation

### Human Evaluation

- Coherence and fluency assessment
- Content preservation evaluation
- Factual consistency verification

## Repository Structure

```
semantic-summarization-system/
│
├── app/                      # Application logic and main entry points
├── api/                      # FastAPI endpoints and request handlers
├── models/                   # Model definitions and architectures
├── pipelines/                # End-to-end inference pipelines
├── datasets/                 # Data loading and preprocessing
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── evaluation/               # Evaluation metrics and assessment tools
├── experiments/              # Experiment configs and results
├── configs/                  # Configuration files (YAML, JSON)
├── docs/                     # Architecture, design, and usage documentation
├── tests/                    # Unit and integration tests
├── scripts/                  # Training, evaluation, and utility scripts
├── utils/                    # Helper functions and utilities
│
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment configuration
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Project Features

- **Modular Architecture**: Clean separation of concerns across models, pipelines, and evaluation
- **Configuration-Driven**: YAML-based configs for reproducible experiments
- **Logging & Monitoring**: Structured logging with loguru for debugging and production monitoring
- **Type Hints**: Full type annotations for code clarity and IDE support
- **Testing Framework**: Pytest-based unit and integration tests
- **Documentation**: Comprehensive docstrings and architectural documentation

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration) or CPU mode
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd semantic-summarization-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Download spaCy model** (optional, for NLP preprocessing)
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Quick Start

### Running the API Server

```bash
python -m api.main
# Server starts at http://localhost:8000
```

### Interactive API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example Usage

```python
from pipelines.summarization import SummarizationPipeline

# Initialize pipeline
pipeline = SummarizationPipeline(model_name="facebook/bart-large-cnn")

# Summarize text
text = "Your long document text here..."
summary = pipeline.summarize(text)
print(summary)
```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black app/ models/ pipelines/ utils/

# Lint code
flake8 app/ models/ pipelines/ utils/

# Type checking
mypy app/ models/ pipelines/
```

### Notebook Development

```bash
jupyter lab notebooks/
```

## Configuration

All configurations are managed through:

1. **Environment Variables** (`.env`)
2. **YAML Config Files** (`configs/`)
3. **Python Config Classes** (Pydantic models)

See [Architecture Documentation](./docs/ARCHITECTURE.md) for detailed configuration options.

## Evaluation

Run evaluation on test datasets:

```bash
python scripts/evaluate.py --model facebook/bart-large-cnn --dataset cnn_dailymail
```

Results are saved to `results/` with detailed metrics breakdown.

## Training (Future)

Fine-tune models on custom datasets:

```bash
python scripts/train.py --config configs/training/bart_base.yaml
```

## Experiments

Track and manage experiments:

```bash
# List all experiments
python scripts/list_experiments.py

# View specific experiment results
python scripts/view_experiment.py --experiment_id <ID>
```

## Future Improvements

- [ ] Multi-lingual summarization support
- [ ] Domain-specific fine-tuning pipelines
- [ ] Real-time model serving with TorchServe
- [ ] Distributed training support
- [ ] Advanced prompt engineering for few-shot learning
- [ ] Retrieval-augmented summarization
- [ ] Factuality verification module
- [ ] Interactive summarization with user feedback loop
- [ ] Mobile-optimized model variants
- [ ] Benchmark comparison dashboard

## Project Structure Decisions

### Why This Architecture?

- **Separation of Concerns**: Models, pipelines, and API are independent
- **Scalability**: Easy to add new models, datasets, and evaluation metrics
- **Reproducibility**: Config-driven experiments ensure consistent results
- **Testing**: Modular structure enables comprehensive test coverage
- **Documentation**: Clear folder organization aids onboarding

### Key Design Principles

1. **Transformer-First**: Leverage pre-trained transformer models
2. **Configuration-Driven**: Minimize hardcoding, maximize flexibility
3. **Type Safety**: Full type hints for better maintainability
4. **Production-Ready**: REST API and monitoring from day one
5. **Experiment-Focused**: Built-in tracking and comparison tools

## Contributing

Contributions are welcome! Please:

1. Create a feature branch
2. Follow code style guidelines (see `.flake8`, `.isort.cfg`)
3. Add tests for new functionality
4. Update documentation

## Citation

If you use this system in your research, please cite:

```bibtex
@software{semantic-summarization-system,
  title={Semantic Summarization System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## References

- [BART Paper](https://arxiv.org/abs/1910.13461)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [PEGASUS Paper](https://arxiv.org/abs/1912.08777)
- [ROUGE Metrics](https://arxiv.org/abs/1602.03606)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)

## License

MIT License - See LICENSE file for details

## Contact

For questions or suggestions, please open an issue on the repository.

---

**Last Updated**: May 2024  
**Version**: 0.1.0 (Initial Architecture)
#   s e m a n t i c - s u m m a r i z a t i o n - s y s t e m  
 