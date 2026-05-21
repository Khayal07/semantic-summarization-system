# CONTRIBUTING

## Contributing Guidelines

We welcome contributions from the community! This document provides guidelines for contributing to the Semantic Summarization System.

## Code of Conduct

Please be respectful and constructive in all interactions. We're committed to providing a welcoming and inclusive environment.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Write/update tests
5. Submit a pull request

## Development Setup

```bash
# Clone repository
git clone <repo-url>
cd semantic-summarization-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

- **Formatting**: Use `black` for code formatting
- **Linting**: Follow `flake8` standards
- **Type Hints**: Use full type annotations
- **Docstrings**: Use Google-style docstrings

```bash
# Format code
black app/ models/ pipelines/ utils/ evaluation/

# Lint code
flake8 app/ models/ pipelines/ utils/ evaluation/

# Type checking
mypy app/ models/ pipelines/ utils/
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=app --cov=models --cov=pipelines
```

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add PEGASUS model support
fix: Handle edge case in text truncation
docs: Update API documentation
refactor: Extract common preprocessing logic
test: Add tests for batch summarization
```

## Pull Request Process

1. Update documentation as needed
2. Add tests for new features
3. Update requirements.txt if dependencies change
4. Ensure all tests pass
5. Request review from maintainers

## Areas for Contribution

- [ ] Multi-lingual summarization support
- [ ] Domain-specific model fine-tuning
- [ ] Performance optimizations
- [ ] Additional evaluation metrics
- [ ] Improved error handling
- [ ] Documentation improvements
- [ ] Example notebooks
- [ ] Containerization (Docker)

## Questions?

Feel free to open an issue for questions or discussions.

---

Thank you for contributing!
