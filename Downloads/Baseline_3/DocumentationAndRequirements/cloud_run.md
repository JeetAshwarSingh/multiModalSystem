# Cloud Training & Evaluation Guide (`engagement-mas`)

This guide provides step-by-step instructions to train and evaluate the **Engagement-MAS** model on Cloud GPUs (**Google Colab**, **Kaggle Notebooks**, or **RunPod / Vast.ai**).

---

## Table of Contents
1. [Prerequisites & Dataset Structure](#1-prerequisites--dataset-structure)
2. [Option 1: Google Colab (Free T4 GPU / Colab Pro)](#option-1-google-colab-free-t4-gpu--colab-pro)
3. [Option 2: Kaggle Code Notebooks (Free 30 hrs/week T4 GPU)](#option-2-kaggle-code-notebooks-free-30-hrsweek-t4-gpu)
4. [Option 3: RunPod / Vast.ai (On-Demand RTX 4090 / A100)](#option-3-runpod--vastai--lambda-labs-on-demand-rtx-4090--a100)
5. [Training & Testing Execution Commands](#5-training--testing-execution-commands)
6. [Retrieving Trained Weights & Results](#6-retrieving-trained-weights--results)

---

## 1. Prerequisites & Dataset Structure

Before uploading to any cloud platform, ensure your project directory has:
* `engagement-mas/` (codebase root)
* `train.csv`, `val.csv`, `test.csv` (DAiSEE CSV split files)
* DAiSEE video dataset folder (`DAiSEE/DataSet/Train`, `DataSet/Validation`, `DataSet/Test`)

---

## Option 1: Google Colab (Free T4 GPU / Colab Pro)

### Step 1: Zip & Upload to Google Drive
1. On your Mac, zip your dataset CSVs and codebase (or mount Google Drive directly).
2. Create a folder in Google Drive named `Engagement_MAS_Project`.
3. Upload `engagement-mas/`, `train.csv`, `val.csv`, `test.csv`, and your `DAiSEE/` video dataset folder to `Engagement_MAS_Project/`.

### Step 2: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2. In the top menu, go to **Runtime > Change runtime type** and select **T4 GPU** (or A100/L4 if using Colab Pro).

### Step 3: Run the Colab Setup Cell
Copy and execute the following code block in your notebook:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to project root in Google Drive
import os
os.chdir('/content/drive/MyDrive/Engagement_MAS_Project/engagement-mas')

# 3. Install requirements
!pip install -r requirements-cloud.txt
```

### Step 4: Verify GPU
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

---

## Option 2: Kaggle Code Notebooks (Free 30 hrs/week T4 GPU)

### Step 1: Create a Kaggle Dataset
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> **New Dataset**.
2. Upload your `train.csv`, `val.csv`, `test.csv` and DAiSEE video dataset folder.
3. Give it a title (e.g., `daisee-dataset`).

### Step 2: Create a Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook**.
2. In the right panel, under **Notebook options**:
   * Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
   * Turn **Internet** to **On**.
3. Under **Input**, click **Add Input** and attach your uploaded `daisee-dataset` and `engagement-mas` codebase repository.

### Step 3: Install & Prepare Code in Notebook Cell
```bash
!pip install -r /kaggle/input/engagement-mas/requirements-cloud.txt
%cd /kaggle/input/engagement-mas
```

---

## Option 3: RunPod / Vast.ai (On-Demand RTX 4090 / A100)

> **Note**: An RTX 4090 or A100 GPU on RunPod (~$0.30–$0.69/hr) will finish Epoch 0 feature extraction in **~15–30 minutes** (compared to 38 hours on Mac CPU).

### Step 1: Deploy Pod
1. Log in to [RunPod.io](https://www.runpod.io).
2. Select **GPU Pods** -> **1x RTX 4090** or **1x A100**.
3. Select the template **PyTorch 2.1+ (CUDA 12.1)**.
4. Click **Deploy**.

### Step 2: Connect & Upload Code
1. Click **Connect** -> **JupyterLab** or **SSH**.
2. Open JupyterLab terminal or SSH from your Mac terminal:
   ```bash
   scp -P <PORT> -r /path/to/engagement-mas root@<RUNPOD_IP>:/workspace/
   scp -P <PORT> -r /path/to/DAiSEE root@<RUNPOD_IP>:/workspace/
   ```

### Step 3: Install Dependencies
```bash
cd /workspace/engagement-mas
pip install -r requirements-cloud.txt
```

---

## 5. Training & Testing Execution Commands

Once setup is complete on Colab, Kaggle, or RunPod:

### Step A: Update Dataset Paths (if needed)
Check `configs/datasets.yaml` and ensure `root_dir`, `train_csv`, `val_csv`, `test_csv` point to your cloud dataset paths:
```yaml
datasets:
  daisee:
    root_dir: "/path/to/DAiSEE"
    train_csv: "/path/to/train.csv"
    val_csv: "/path/to/val.csv"
    test_csv: "/path/to/test.csv"
```

### Step B: Launch Training
Run the training script with GPU acceleration enabled (`device: "auto"`):

```bash
python -m training.train_macro \
  --config configs/training.yaml \
  --dataset daisee \
  --split train
```

### Step C: Run Evaluation across Fusion Methods
After training completes, evaluate your trained weights (`weights/macro/best.pt` or `last.pt`):

1. **Static Late Fusion**:
   ```bash
   python -m evaluation.evaluate \
     --config configs/fusion.yaml \
     --dataset daisee \
     --split test \
     --fusion static
   ```

2. **Agentic Fusion**:
   ```bash
   python -m evaluation.evaluate \
     --config configs/fusion.yaml \
     --dataset daisee \
     --split test \
     --fusion agentic
   ```

3. **Early Fusion**:
   ```bash
   python -m evaluation.evaluate \
     --config configs/fusion.yaml \
     --dataset daisee \
     --split test \
     --fusion early
   ```

---

## 6. Retrieving Trained Weights & Results

Once training and evaluation finish on the cloud:

* **Trained Weights Location**: `weights/macro/best.pt` and `weights/macro/last.pt`
* **Evaluation Results**: `evaluation/results/`

### Downloading back to your Mac:
* **Google Colab**: Download `weights/macro/best.pt` from the left sidebar File Browser or Google Drive folder.
* **Kaggle**: Use `File -> Download File` or save outputs to `/kaggle/working`.
* **RunPod (via SCP on local Mac)**:
  ```bash
  scp -P <PORT> root@<RUNPOD_IP>:/workspace/engagement-mas/weights/macro/best.pt ./weights/macro/
  ```
