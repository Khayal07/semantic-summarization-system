# Quick Reference Guide

## Project Overview

**Semantic Summarization System** - A production-grade NLP system for abstractive and extractive text summarization using transformer models (BART, T5, PEGASUS).

## Quick Start

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Run API
```bash
python -m api.main
# http://localhost:8000/docs
```

### Generate Summary
```bash
python scripts/inference.py --input-file document.txt
```

### Evaluate Model
```bash
python scripts/evaluate.py --model facebook/bart-large-cnn --dataset cnn_dailymail
```

## Project Structure

```
semantic-summarization-system/
├── app/              # Application entry point
├── api/              # FastAPI REST endpoints
├── models/           # Model implementations
├── pipelines/        # Inference pipelines
├── datasets/         # Data loading utilities
├── evaluation/       # Evaluation metrics
├── configs/          # Configuration files
├── docs/             # Documentation
├── tests/            # Unit tests
├── scripts/          # CLI utilities
├── utils/            # Helper functions
├── notebooks/        # Jupyter notebooks
├── experiments/      # Experiment tracking
└── requirements.txt  # Dependencies
```

## Key Features

- **Abstractive Summarization**: BART, T5, PEGASUS models
- **Extractive Summarization**: Sentence ranking approaches
- **REST API**: FastAPI with async support
- **Evaluation**: ROUGE, BERTScore, BLEU metrics
- **Configuration-Driven**: YAML-based configuration
- **Production-Ready**: Type hints, logging, error handling

## Core Modules

### Models (`models/`)
- `base.py`: Abstract base class
- `abstractive.py`: Seq2seq models
- `extractive.py`: Sentence ranking

### Pipelines (`pipelines/`)
- `summarization.py`: End-to-end orchestration

### API (`api/`)
- `main.py`: FastAPI application
- `/api/v1/summarize`: Generate summaries
- `/api/v1/models`: List available models
- `/health`: Health check

### Evaluation (`evaluation/`)
- `metrics.py`: ROUGE, BERTScore, BLEU

### Utils (`utils/`)
- `config.py`: Configuration management
- `text_utils.py`: Text processing

## Configuration

Edit `.env` for:
- Model selection
- API settings
- Device (GPU/CPU)
- Paths

Edit `configs/` for:
- Model specifications
- Inference parameters
- Evaluation settings

## API Usage

### Python Client
```python
from pipelines.summarization import SummarizationPipeline

pipeline = SummarizationPipeline(model_name="facebook/bart-large-cnn")
pipeline.load_model()
summary = pipeline.summarize("Your text here...")
```

### HTTP Request
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "max_length": 128,
    "min_length": 30
  }'
```

## Development

### Code Quality
```bash
black app/ models/ pipelines/ utils/
flake8 app/ models/ pipelines/ utils/
mypy app/ models/ pipelines/ utils/
```

### Testing
```bash
pytest tests/ -v
pytest tests/test_pipeline.py -v
```

### Type Checking
All functions use type hints for clarity and IDE support.

## Available Models

- `facebook/bart-large-cnn` - BART fine-tuned on CNN/DailyMail
- `google/t5-base` - T5 base model
- `google/pegasus-large` - PEGASUS specialized for summarization

## Evaluation Metrics

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Semantic similarity
- **Compression Ratio**: Output/Input length ratio

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU out of memory | Reduce batch size, use smaller model |
| Model not found | Check `MODEL_NAME` in `.env` |
| CUDA unavailable | Set `DEVICE=cpu` in `.env` |
| API not responding | Check port 8000 is available |

## Documentation Files

- `README.md` - Full project overview
- `ARCHITECTURE.md` - System design details
- `SETUP.md` - Installation guide
- `CONTRIBUTING.md` - Contribution guidelines
- `DEVELOPMENT.md` - Development notes

## Performance Tips

1. **Model Loading**: Models cached after first load
2. **Batch Processing**: Process multiple documents together
3. **Mixed Precision**: Use FP16 for faster inference
4. **Beam Search**: Adjust `num_beams` for speed/quality tradeoff

## Next Steps

1. [ ] Review architecture in `docs/ARCHITECTURE.md`
2. [ ] Run setup in `SETUP.md`
3. [ ] Try examples in quick start
4. [ ] Explore notebooks in `notebooks/`
5. [ ] Run evaluation script
6. [ ] Check contribution guidelines

## Resources

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ROUGE Metrics](https://en.wikipedia.org/wiki/ROUGE_(metric))

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: See `docs/` directory
- Examples: Check `notebooks/` for tutorials

---

**Version**: 0.1.0  
**Last Updated**: May 2024
