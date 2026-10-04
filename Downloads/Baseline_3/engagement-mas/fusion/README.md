# Fusion Overview

The **fusion** package implements three strategies for combining the outputs of the perception blocks (Macro, rPPG, MER) after they have been synchronized by the `SyncGate`.

1. **Early‑Fusion Baseline** (`early_fusion.py`)
   * Concatenates the *feature vectors* from all modalities that are present in a bundle.
   * Missing modalities are masked (the corresponding slice is omitted, not zero‑filled).
   * A shallow trainable head (single linear layer) maps the concatenated vector to the engagement classes.
   * This head is trained on the same DAiSEE labels as the Macro block.

2. **Static Late‑Fusion** (`static_late_fusion.py`)
   * Operates on *per‑modality predictions* (class probability vectors) that the gate already provides.
   * A set of **static weights** (configurable in `configs/fusion.yaml`) is used to compute a weighted average of the predictions.
   * No learnable parameters – deterministic and extremely lightweight.

3. **Agentic Late‑Fusion** (`agentic_fusion.py`)
   * Introduces three agents that act on the bundle after the gate release:
     - **Modality Monitor** – extracts a reliability score from each modality’s quality metrics.
     - **Reasoning Engine** – adjusts the modality weights based on reliability using deterministic rule‑based thresholds (configurable). It also records a human‑readable explanation for each adjustment.
     - **State Controller** – holds the current confidence scores and a short‑term memory buffer (last N bundles) to allow temporal smoothing.
   * The final fused engagement prediction is obtained by applying the adjusted weights to the modality predictions.
   * All agents are deterministic by default; an optional LLM backend (e.g., Ollama) can be swapped in later but is **not required**.

Each fusion method follows the common block interface and returns a dictionary
```json
{
  "clip_id": "...",
  "modality": "fusion",
  "predictions": {"engagement": [p0, p1, p2, p3]},
  "explanation": "... (only for agentic fusion)",
  "timestamps": [...],
  "status": "success"
}
```

All code is written to be fully deterministic (fixed RNG seeds, no stochastic layers) and can be run on the Mac (CPU/MPS) or on the cloud GPU without modification.
