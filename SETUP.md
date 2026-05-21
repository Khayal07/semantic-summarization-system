# Setup Guide

## System Requirements

- **Python**: 3.9 or higher
- **CUDA**: 11.8+ (recommended for GPU support)
- **RAM**: 16GB minimum (32GB recommended for large models)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/semantic-summarization-system.git
cd semantic-summarization-system
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n summarization python=3.10
conda activate summarization
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

The system will automatically download models on first use. To pre-download:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'facebook/bart-large-cnn'
AutoTokenizer.from_pretrained(model_name)
AutoModelForSeq2SeqLM.from_pretrained(model_name)
"
```

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Model Configuration
MODEL_NAME=facebook/bart-large-cnn
DEVICE=cuda
MAX_LENGTH=1024

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Paths
DATASET_PATH=./datasets
MODEL_CACHE_PATH=./model_cache
```

### 6. Download spaCy Model (Optional)

```bash
python -m spacy download en_core_web_sm
```

## Running the System

### API Server

```bash
python -m api.main
```

Server will start at `http://localhost:8000`

Visit http://localhost:8000/docs for interactive API documentation.

### Command Line Inference

```bash
# Summarize from file
python scripts/inference.py --input-file input.txt --output-file summary.txt

# Summarize from stdin
echo "Your text here..." | python scripts/inference.py

# Custom model and parameters
python scripts/inference.py --model google/t5-base --max-length 150
```

### Evaluation

```bash
# Evaluate on CNN/DailyMail
python scripts/evaluate.py --model facebook/bart-large-cnn --dataset cnn_dailymail

# Evaluate on custom dataset
python scripts/evaluate.py --model facebook/bart-large-cnn --max_samples 500
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter GPU out-of-memory errors:

1. Reduce batch size in config
2. Reduce `max_length` parameter
3. Use a smaller model (e.g., `facebook/bart-base`)
4. Enable mixed precision training

### Model Download Issues

If models fail to download:

```bash
# Set HF token
huggingface-cli login

# Or set token in environment
export HF_TOKEN=your_token_here
```

### CUDA Not Available

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU mode
export DEVICE=cpu
```

## Development Setup

```bash
# Install development dependencies
pip install pytest black flake8 mypy jupyter

# Format code
black app/ models/ pipelines/ utils/

# Run tests
pytest tests/ -v

# Start Jupyter
jupyter lab notebooks/
```

## Docker Setup (Optional)

```bash
docker build -t semantic-summarization .
docker run -p 8000:8000 semantic-summarization
```

## Next Steps

1. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
2. Check [README.md](README.md) for usage examples
3. Explore notebooks in `notebooks/` for tutorials
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## Getting Help

- Check issues on GitHub
- Review documentation in `docs/`
- Look at example notebooks
- Open a new issue for bugs or questions
