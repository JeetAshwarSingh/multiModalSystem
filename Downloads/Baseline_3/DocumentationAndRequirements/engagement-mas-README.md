# Engagement-MAS

**Engagement-MAS** is a multi‑agent system for estimating student engagement from video clips. It consists of three parallel perception blocks (Macro‑vision, rPPG, Micro‑Expression), a synchronization gate, and three fusion strategies (early, static late, agentic late). The system is fully modular, deterministic for non‑trained components, and supports training the Macro block on a free cloud GPU.

---

## Architecture Diagram

```mermaid
flowchart LR
    subgraph Perception
        macro[Macro Block (Vision) ]
        rppg[rPPG Block]
        mer[Micro‑Expression Block]
    end
    macro -->|features, predictions| gate
    rppg -->|features, predictions| gate
    mer -->|features, predictions| gate
    gate -->|bundle| fusion[Fusion Layer]
    fusion -->|final engagement label| output[Output]
    classDef block fill:#f9f,stroke:#333,stroke-width:2px;
    class macro,rppg,mer,gate,fusion block;
```

---

## Repository Layout

```
engagement-mas/
├─ env/           # Conda environment files
├─ configs/       # YAML configuration files
├─ data/          # Dataset loaders and schema definitions
├─ blocks/        # Per‑modality processing blocks
│   ├─ macro/    # Trained vision block (uses DINOv2)
│   ├─ rppg/     # Deterministic rPPG wrapper (Face2PPG)
│   └─ mer/      # Deterministic micro‑expression block
├─ gate/          # Synchronization gate implementation
├─ fusion/        # Early, static late & agentic late fusion
├─ agents/        # Modality Monitor, Reasoning Engine, State Controller
├─ training/      # Cloud training scripts & notebooks
├─ weights/       # Model checkpoints and pretrained assets
├─ evaluation/    # Metrics, ablations, latency benchmarks
└─ tests/         # Unit and integration tests
```

Each sub‑directory contains a `README.md` that explains its purpose, inputs/outputs, and how to run the code (see the individual README files).

---

## Getting Started

1. **Create the appropriate Conda environment** (see `env/README.md`).
2. **Prepare the dataset** (see `data/README.md`).
3. **Run the perception blocks** individually or together through the gate (`gate/README.md`).
4. **Choose a fusion strategy** (`fusion/README.md`).
5. **Train the Macro block** on a free cloud GPU (`training/README.md`).

Feel free to explore the `tests/` folder for reproducibility checks.
