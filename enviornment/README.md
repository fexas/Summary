# Environment Setup Guide

This folder contains configuration files to replicate the development environment on another machine (specifically macOS).

## Option 1: Using Conda (Recommended)

If you have Anaconda or Miniconda installed, you can create a new environment directly from the `environment.yml` file. This is the most reliable method as it handles non-Python dependencies as well.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate <env_name>
```

*Note: You may need to edit the `prefix` line at the end of `environment.yml` if the username or path differs on the new machine.*

## Option 2: Using Pip

If you prefer using `pip` or standard Python virtual environments:

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Note for Apple Silicon (M1/M2/M3) users

This environment is configured on macOS. Some packages (like `tensorflow`, `torch` with MPS support) might have platform-specific binaries. If you encounter issues on a different architecture (e.g., Linux or Intel Mac), you might need to relax some version constraints in `requirements.txt` or let Conda resolve dependencies afresh.
