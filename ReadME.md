# Neural Estimation of Summary Statistics for Amortized Inference with Non-asymptotic Guarantees — Code

This repository accompanies the paper **“Neural Estimation of Summary Statistics for Amortized Inference with Non-asymptotic Guarantees”**. It contains experiment code for three benchmarks:

- **Gaussian**
- **Lotka–Volterra**
- **SIR**

Each benchmark folder provides the corresponding model implementations, a runnable experiment entry script, and shared helper utilities. Additional scripts (when present) are mainly used to reproduce figures or diagnostics shown in the paper.

## Repository Structure

At a high level, the repository is organized as:

```
Summary/
  Gaussian/
  LotkaVoletrra/
  SIR/
  enviornment/
```

Within each benchmark folder (e.g., `Gaussian/`, `LotkaVoletrra/`, `SIR/`), you will typically find:

- `models/`: model definitions used in that benchmark
- `run_experiment.py`: main entry point for running experiments
- `utilities.py`: helper functions (metrics, refinement, plotting helpers, etc.)
- `config.json`: experiment configuration (where applicable)

## Notes

- Some extra scripts and subfolders may exist to support specific plotting or analysis tasks related to the paper’s figures.
- Environment snapshots are stored under `enviornment/` (note the folder name spelling).
