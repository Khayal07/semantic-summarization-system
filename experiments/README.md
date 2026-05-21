# Experiments

This directory contains configurations and results for experiments conducted with the system.

## Directory Structure

```
experiments/
├── configs/          # Experiment configuration files
│   ├── exp001_bart_baseline.yaml
│   ├── exp002_t5_comparison.yaml
│   └── exp003_pegasus_finetuned.yaml
├── results/          # Experiment results
│   ├── exp001_results.json
│   ├── exp002_results.json
│   └── exp003_results.json
└── README.md        # This file
```

## Running Experiments

1. Create experiment config in `configs/`
2. Run evaluation or training:
   ```bash
   python scripts/evaluate.py --model facebook/bart-large-cnn
   ```
3. Results automatically saved to `results/`

## Experiment Tracking

Each experiment should include:
- Model name and version
- Dataset used
- Hyperparameters
- Metrics (ROUGE, BERTScore, etc.)
- Date and time
- Notes and observations

## Comparing Results

Use provided comparison scripts:
```bash
python scripts/compare_experiments.py results/exp001 results/exp002
```
