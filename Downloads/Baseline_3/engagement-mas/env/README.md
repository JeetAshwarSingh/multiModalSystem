# Environment Setup

This directory contains two Conda environment specifications:

* **`environment-mac.yml`** – for development and inference on your MacBook M1 (osx‑arm64). Uses the CPU/MPS backend; **no CUDA** packages are included.
* **`environment-cloud.yml`** – for training on a free cloud GPU (Linux x86_64 with CUDA). Includes the CUDA toolkit and GPU‑enabled PyTorch.

Both files pin the versions of all dependencies to ensure reproducibility. After creating the environment you can activate it with:

```bash
# Mac (CPU/MPS)
conda env create -f env/environment-mac.yml
conda activate engagement-mas-mac

# Cloud (CUDA)
conda env create -f env/environment-cloud.yml
conda activate engagement-mas-cloud
```

> **Tip:** If you already have a base Conda installation, you may need to run `conda install -c conda-forge mamba` first and then use `mamba env create …` for faster solves.

The rest of the repository assumes the active environment contains the packages listed in the respective yaml files.
